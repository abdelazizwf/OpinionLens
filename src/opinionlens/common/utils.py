import logging
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    filename: str | None = None
) -> logging.Logger:
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date_fmt,
        filename=filename,
        filemode="w"
    )
    return logging.getLogger(name)


def get_timestamp() -> str:
    time = datetime.now().replace(microsecond=0).isoformat()
    replacements = str.maketrans("", "", "T:-")
    time = time.translate(replacements)
    return time
