# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator


@contextlib.contextmanager
def suppress_stdout(suppress: bool = True) -> Iterator[None]:
    with open(os.devnull, mode="w") as devnull:
        stdout = sys.stdout
        if suppress:
            sys.stdout = devnull
        try:
            yield
        finally:
            if suppress:
                sys.stdout = stdout
