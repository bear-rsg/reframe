import math
import sys
import time
import threading
import queue
from datetime import datetime
from queue import Queue

from reframe.core.exceptions import TaskExit
from reframe.core.logging import getlogger
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)


class SerialExecutionPolicy(ExecutionPolicy):
    def __init__(self):
        super().__init__()
        self._tasks = []

    def run_check(self, check, partition, environ):
        super().run_check(check, partition, environ)
        self.printer.status(
            'RUN', "%s on %s using %s" %
            (check.name, partition.fullname, environ.name)
        )
        task = RegressionTask(check)
        self._tasks.append(task)
        self.stats.add_task(task)
        try:
            task.setup(partition, environ,
                       sched_account=self.sched_account,
                       sched_partition=self.sched_partition,
                       sched_reservation=self.sched_reservation,
                       sched_nodelist=self.sched_nodelist,
                       sched_exclude_nodelist=self.sched_exclude_nodelist,
                       sched_options=self.sched_options)

            task.compile()
            task.run()
            task.wait()
            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            if task.failed:
                remove_stage_files = False
            else:
                remove_stage_files = not self.keep_stage_files

            task.cleanup(not self.keep_stage_files, False)

        except TaskExit:
            return
        except ABORT_REASONS as e:
            task.abort(e)
            raise
        except BaseException:
            task.fail(sys.exc_info())
        finally:
            self.printer.status('FAIL' if task.failed else 'OK',
                                task.check.info(), just='right')


class PollRateFunction:
    def __init__(self, min_rate, decay_time):
        self._min_rate = min_rate
        self._decay = decay_time
        self._thres = 0.05

        # decay function parameters
        self._a = None
        self._b = None
        self._c = None

    def _init_poll_fn(self, init_rate):
        self._init_rate = init_rate
        self._b = self._min_rate
        log_arg = (init_rate - self._b) / (self._thres*self._b)
        if log_arg < sys.float_info.min:
            self._a = 0.0
            self._c = 0.0
        else:
            self._a = init_rate - self._b
            self._c = math.log(self._a / (self._thres*self._b)) / self._decay

        getlogger().debug('rate equation: %.3f*exp(-%.3f*x)+%.3f' %
                          (self._a, self._c, self._b))

    def __call__(self, x, init_rate):
        if self._a is None:
            self._init_poll_fn(init_rate)

        return self._a*math.exp(-self._c*x) + self._b


class TaskFinalizerThread(threading.Thread):
    def __init__(self, policy):

        super().__init__()

        self.daemon = True
        
        self._policy = policy
        self._retired_tasks = Queue()
        self._current_task = None

        self._stop_event = threading.Event()
        self._stop_when_done_event = threading.Event()

    def request_stop(self):
        """Requests that the thread stop finalizing tasks."""
        self._stop_event.set()

    def request_stop_when_done(self):
        """Requests that the thread stops when the retired tasks queue is empty."""
        self._stop_when_done_event.set()

    def finalize_tasks(self):
        """Waits for the thread to process all of the events in the queue, then
        returns.
        """
        self.request_stop_when_done()
        self.join()

    def should_stop(self):
        """Returns whether or not the thread has been requested to stop."""
        return self._stop_event.is_set()

    def abort_retired(self, cause):
        """Abort all currently retired tasks, with a given cause."""
        # Stop thread
        self.request_stop()
        
        # Abort remaining tasks
        while True:
            try:
                task = self._retired_tasks.get_nowait()
            except queue.Empty:
                break
            
            task.abort(cause)
        
        # Give thread 5 seconds to exit.
        # Since the thread is a daemon, when the python process exits, the thread should
        # abort.
        # This will stop some cleanup code from running for the task that the thread is
        # currently processing, however this is fine as it doesn't affect slurm -- all the
        # running tasks have been aborted already -- this thread is just for already
        # finished tasks.
        getlogger().debug('aborting: joining finalizer thread with timeout 5 seconds...')
        self.join(timeout=5.0)
        
        if self._current_task is not None:
            # Abort the currently finalizing task.
            self._current_task.abort(cause)

    def _finalize_current_task(self):
        if not self._policy.skip_sanity_check:
            self._current_task.sanity()

        if not self._policy.skip_performance_check:
            self._current_task.performance()

        self._current_task.cleanup(not self._policy.keep_stage_files, False)
        self._current_task = None

    def num_retired_tasks(self):
        """Returns the number of tasks still to be finalized."""
        return self._retired_tasks.qsize()

    def retire_task(self, task):
        """Adds a task to the retiring queue."""
        getlogger().debug('retiring task: %s' % task.check.info())
        getlogger().debug('retired queue: %d task(s)' % self.num_retired_tasks())
        self._retired_tasks.put(task)

    def run(self):
        try:
            while not self.should_stop():
                n = self.num_retired_tasks()
                if n:
                    getlogger().debug('retired queue: %d task(s)' % n)
                try:
                    self._current_task = self._retired_tasks.get(True, timeout=0.5)
                except queue.Empty:
                    if self._stop_when_done_event.is_set():
                        break  # Break -- no more tasks to come
                    else:
                        continue  # Retry

                # Finalize task
                getlogger().debug('finalizing task: %s' % self._current_task.check.info())
                try:
                    self._finalize_current_task()
                except TaskExit:
                    pass

        finally:
            getlogger().debug('finalizer thread: exiting')

class AsynchronousExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):

        super().__init__()

        # All currently running tasks
        self._running_tasks = []
        
        # Task finalizer thread
        self._finalizer_thread = TaskFinalizerThread(self)
        self._finalizer_thread.start()

        # Counts of running tasks per partition
        self._running_tasks_counts = {}

        # Ready tasks to be executed per partition
        self._ready_tasks = {}

        # The tasks associated with the running checks
        self._tasks = []

        # Job limit per partition
        self._max_jobs = {}

        self.task_listeners.append(self)

    def _remove_from_running(self, task):
        getlogger().debug('removing task: %s' % task.check.info())
        try:
            self._running_tasks.remove(task)
        except ValueError:
            getlogger().debug('not in running tasks')
            pass
        else:
            partname = task.check.current_partition.fullname
            self._running_tasks_counts[partname] -= 1

    def on_task_run(self, task):
        partname = task.check.current_partition.fullname
        self._running_tasks_counts[partname] += 1
        self._running_tasks.append(task)

    def on_task_failure(self, task):
        self._remove_from_running(task)
        self.printer.status('FAIL', task.check.info(), just='right')

    def on_task_success(self, task):
        self.printer.status('OK', task.check.info(), just='right')

    def on_task_exit(self, task):
        task.wait()
        self._remove_from_running(task)
        self._finalizer_thread.retire_task(task)

    def enter_partition(self, c, p):
        self._running_tasks_counts.setdefault(p.fullname, 0)
        self._ready_tasks.setdefault(p.fullname, [])
        self._max_jobs.setdefault(p.fullname, p.max_jobs)

    def run_check(self, check, partition, environ):
        super().run_check(check, partition, environ)
        task = RegressionTask(check, self.task_listeners)
        self._tasks.append(task)
        self.stats.add_task(task)
        try:
            task.setup(partition, environ,
                       sched_account=self.sched_account,
                       sched_partition=self.sched_partition,
                       sched_reservation=self.sched_reservation,
                       sched_nodelist=self.sched_nodelist,
                       sched_exclude_nodelist=self.sched_exclude_nodelist,
                       sched_options=self.sched_options)

            self.printer.status('RUN', task.check.info())
            partname = partition.fullname
            if self._running_tasks_counts[partname] >= partition.max_jobs:
                # Make sure that we still exceeded the job limit
                getlogger().debug('reached job limit (%s) for partition %s' %
                                  (partition.max_jobs, partname))
                self._poll_tasks()

            if self._running_tasks_counts[partname] < partition.max_jobs:
                # Test's environment is already loaded; no need to be reloaded
                self._reschedule(task, load_env=False)
            else:
                self.printer.status('HOLD', task.check.info(), just='right')
                self._ready_tasks[partname].append(task)
        except TaskExit:
            return
        except ABORT_REASONS as e:
            if not task.failed:
                # Abort was caused due to failure elsewhere, abort current
                # task as well
                task.abort(e)

            self._failall(e)
            raise

    def _poll_tasks(self):
        """Update the counts of running checks per partition."""
        getlogger().debug('updating counts for running test cases')
        getlogger().debug('polling %s task(s)' % len(self._running_tasks))
        for t in self._running_tasks:
            t.poll()

    def _failall(self, cause):
        """Mark all tests as failures"""
        try:
            while True:
                self._running_tasks.pop().abort(cause)
        except IndexError:
            pass

        for ready_list in self._ready_tasks.values():
            getlogger().debug('ready list size: %s' % len(ready_list))
            for task in ready_list:
                task.abort(cause)

        self._finalizer_thread.abort_retired(cause)

    def _reschedule(self, task, load_env=True):
        getlogger().debug('scheduling test case for running')
        partname = task.check.current_partition.fullname

        # Restore the test case's environment and run it
        if load_env:
            task.resume()

        task.compile()
        task.run()

    def _reschedule_all(self):
        for partname, num_jobs in self._running_tasks_counts.items():
            assert(num_jobs >= 0)
            num_empty_slots = self._max_jobs[partname] - num_jobs
            num_rescheduled = 0
            for i in range(num_empty_slots):
                try:
                    task = self._ready_tasks[partname].pop()
                except IndexError:
                    break

                self._reschedule(task)
                num_rescheduled += 1

            if num_rescheduled:
                getlogger().debug('rescheduled %s job(s) on %s' %
                                  (num_rescheduled, partname))

    def exit(self):
        self.printer.separator('short single line',
                               'waiting for spawned checks to finish')
        pollrate = PollRateFunction(0.2, 60)
        num_polls = 0
        t_start = datetime.now()
        try:
            while self._running_tasks:
                getlogger().debug('running tasks: %s' % len(self._running_tasks))
                num_polls += len(self._running_tasks)
                try:
                    self._poll_tasks()
                    self._reschedule_all()
                    t_elapsed = (datetime.now() - t_start).total_seconds()
                    real_rate = num_polls / t_elapsed
                    getlogger().debug(
                        'polling rate (real): %.3f polls/sec' % real_rate)

                    if len(self._running_tasks):
                        desired_rate = pollrate(t_elapsed, real_rate)
                        getlogger().debug(
                            'polling rate (desired): %.3f' % desired_rate)
                        t = len(self._running_tasks) / desired_rate
                        getlogger().debug('sleeping: %.3fs' % t)
                        time.sleep(t)
                except TaskExit:
                    pass

            getlogger().debug('run all tasks. finalizing tasks...')
            self._finalizer_thread.finalize_tasks()

        except ABORT_REASONS as e:
            self._failall(e)
            raise

        self.printer.separator('short single line',
                               'all spawned checks have finished\n')
