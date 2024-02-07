from functools import wraps
from time import perf_counter
from typing import Callable


def time_func(func: Callable) -> Callable:
    """Add timing wrapper to function.

    Wrapped function returns tuple:
        tuple[0] = Duration of function in seconds.
        tuple[1] = Returned result from function.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start = perf_counter()

        result = func(*args, **kwargs)
        end = perf_counter()
        duration = end - start

        return (duration, result)

    return wrapped_func
