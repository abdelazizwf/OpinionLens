import logging
import os
from datetime import datetime


def get_csv_files(path: str, prefix: str | None = None) -> list[str]:
    assert os.path.exists(path), f"{path!r} doesn't exist!"

    if os.path.isfile(path):
        return [path]

    paths = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.split(".")[-1] == "csv":
                if not prefix or filename.startswith(prefix):
                    paths.append(os.path.join(dirname, filename))

    return paths


def get_logger(
    name: str,
    level: int = logging.INFO,
    filename: str | None = None
) -> logging.Logger:
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a+")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    time = datetime.now().replace(microsecond=0).isoformat()
    replacements = str.maketrans("", "", "T:-")
    time = time.translate(replacements)
    return time
