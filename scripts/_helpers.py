import time
import pandas as pd
from pathlib import Path
from functools import partial
from requests.exceptions import HTTPError


def to_date_period(dt):
    assert dt.tz is not None, 'Datetime must have timezone'
    dt = dt.tz_convert('Europe/London')
    return dt.strftime('%Y-%m-%d'), dt.hour * 2 + dt.minute // 30 + 1


def to_datetime(day, period):
    return pd.Timestamp(day, tz='Europe/London').tz_convert('utc') + pd.Timedelta(minutes=30) * (period - 1)


def to_daterange(*args):
    '''Somewhat redundant, but ensures consistency between datasets'''

    if len(args) == 2:
        assert (start.tz is not None + end.tz is not None) == 2, 'Both start and end must be timezone-naive'
        start, end = args[0], args[1]
    
    elif len(args) == 1:

        if not isinstance(args[0], pd.Timestamp):
            assert len(args[0]) == 10, 'Expects single day or start and end dates'

        start, end = pd.Timestamp(args[0]), pd.Timestamp(args[0]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)

    return pd.date_range(start, end, freq='30min', tz='Europe/London')


def get_run_path(fn, dir, rdir, shared_resources, exclude_from_shared):
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.
    exclude_from_shared: list
        List of filenames to exclude from shared resources. Only relevant if shared_resources is "base".

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """
    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        not_shared_rule = (
            not fn.startswith("networks/elec")
            and not fn.startswith("add_electricity")
            and not any(fn.startswith(ex) for ex in exclude_from_shared)
        )
        is_shared = no_relevant_wildcards and not_shared_rule
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling."
        )

    return f"{dir}{rdir}{fn}"


def path_provider(dir, rdir, shared_resources, exclude_from_shared):
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(
        get_run_path,
        dir=dir,
        rdir=rdir,
        shared_resources=shared_resources,
        exclude_from_shared=exclude_from_shared,
    )


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    import logging
    import sys

    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..", "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)

    # Setup a function to handle uncaught exceptions and include them with their stacktrace into logfiles
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception