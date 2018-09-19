from pathos.multiprocessing import ProcessingPool as Pool

def parallel_map(func, array, n_workers=3):
    # workaround to enable interrupting with CTRL+C
    def interruptable_func(i):
        try:
            return func(i)
        except KeyboardInterrupt:
            raise RuntimeError("Keyboard interrupt")
    return Pool(n_workers).map(interruptable_func, array)
