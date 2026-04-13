import logging


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


def get_tqdm():
    """Returns the appropriate tqdm implementation based on the current environment.

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
