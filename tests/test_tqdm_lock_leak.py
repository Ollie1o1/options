"""The screener leaked a POSIX semaphore on every exit.

Observed on a real run:

    resource_tracker: There appear to be 1 leaked semaphore objects to clean up
    at shutdown: {'/mp-wfgou9pk'}

The source is tqdm, not this codebase — `tqdm.get_lock()` lazily builds a
`TqdmDefaultWriteLock` whose first member is a `multiprocessing.synchronize.RLock`,
so it allocates a kernel semaphore the moment the first progress bar is created.

That lock only earns its keep when bars are driven from separate *processes*.
Every bar here is driven from one process (the fetch pool is a
ThreadPoolExecutor), so a plain thread lock is the correct primitive and leaves
no semaphore to leak.
"""
import unittest


def _lock_module_names(lock):
    """Module names of the primitives inside a tqdm lock, however it's shaped."""
    members = getattr(lock, "locks", None)
    if members is None:
        members = [lock]
    return [type(m).__module__ for m in members]


class TqdmLockTest(unittest.TestCase):
    def test_screener_import_leaves_tqdm_without_a_multiprocessing_lock(self):
        import src.options_screener  # noqa: F401  (installs the thread lock)
        from tqdm import tqdm
        self.assertNotIn("multiprocessing.synchronize",
                         _lock_module_names(tqdm.get_lock()))

    def test_a_real_progress_bar_still_works_under_the_thread_lock(self):
        # Swapping the lock must not break the bar it guards.
        import io

        import src.options_screener  # noqa: F401
        from tqdm import tqdm
        sink = io.StringIO()
        bar = tqdm(total=2, file=sink, leave=False)
        bar.update(1)
        bar.update(1)
        bar.close()
        self.assertNotIn("multiprocessing.synchronize",
                         _lock_module_names(tqdm.get_lock()))

    def test_the_lock_is_still_usable_as_a_context_manager(self):
        import src.options_screener  # noqa: F401
        from tqdm import tqdm
        with tqdm.get_lock():
            pass


if __name__ == "__main__":
    unittest.main()
