# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import Optional, Union


def _setup_handlers(logger, handlers=None, reset=True):
    if reset:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    if handlers:
        for handler in handlers:
            logger.addHandler(handler)


def _setup_formaters(logger, format_):
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter(format_))


def _setup_logger(logger, level, format_, handlers):
    logger.setLevel(level)
    _setup_handlers(logger, handlers, reset=True)
    _setup_formaters(logger, format_)


#  pylint: disable=redefined-builtin
def setup_logger(
    logger: Optional[logging.Logger] = None,
    level: Optional[Union[int, str]] = logging.INFO,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    force: bool = True,
    stream: bool = True,
    filename: Optional[Union[str, bytes, os.PathLike]] = None,
    **kwargs,
) -> None:
    handlers = []
    if stream:
        handlers += [logging.StreamHandler()]

    if filename:
        handlers += [logging.FileHandler(filename)]

    if logger is not None:
        _setup_logger(logger, level, format, handlers)
        return

    kwargs = {"level": level, "format": format, "handlers": handlers}
    if force:
        if sys.version_info < (3, 8):
            _setup_handlers(logging.root, reset=True)
        else:
            kwargs["force"] = force

    logging.basicConfig(**kwargs)


def log_dict(_logger: logging.Logger, d: Mapping, sep: str = " = ") -> None:
    for key, value in d.items():
        _logger.info(f"{key}{sep}{value}")
