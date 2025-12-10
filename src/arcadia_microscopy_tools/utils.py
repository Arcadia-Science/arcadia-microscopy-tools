import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any


def configure_logging(verbose):
    """Configure the Python logging system with optional verbosity.

    Sets up a basic logging configuration with a standardized format for timestamps,
    logger names, and log levels. The verbosity level controls whether DEBUG messages
    are displayed.

    Args:
        verbose:
            If True, sets logging level to DEBUG to show all messages.
            If False, sets logging level to INFO which filters out DEBUG messages.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parallelized(
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """Decorator to run a dataclass instance method concurrently over an iterable.

    The decorated method must accept *one* item of work as its first argument. When you call the
    method, pass an **iterable** of such items. The decorator fans out calls to each item using
    a ThreadPoolExecutor and returns a list of results in the same order as the input.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(items: Iterable[Any], *args, **kwargs) -> list[Any]:
            # Convert to list so we can preserve order later
            _items = list(items)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map each future back to its position to keep results ordered
                future_to_index = {
                    executor.submit(func, item, *args, **kwargs): idx
                    for idx, item in enumerate(_items)
                }
                results: list[Any] = [None] * len(_items)

                if show_progress:
                    tqdm = get_tqdm()
                    futures = tqdm(as_completed(future_to_index), total=len(results))
                else:
                    futures = as_completed(future_to_index)

                for future in futures:
                    idx = future_to_index[future]
                    results[idx] = future.result()
            return results

        return wrapper

    return decorator


def get_tqdm():
    """
    Returns the appropriate tqdm implementation based on the current environment.

    Returns:
        The tqdm implementation suitable for the current environment:
        - tqdm.notebook.tqdm for Jupyter/IPython notebook environments
        - tqdm.tqdm for standard environments
    """
    try:
        # Check if inside a notebook environment
        get_ipython().__class__.__name__  # type: ignore # noqa: B018
        from tqdm.notebook import tqdm
    except NameError:
        from tqdm import tqdm

    return tqdm
