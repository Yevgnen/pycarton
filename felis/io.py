# -*- coding: utf-8 -*-

import contextlib
import os
import subprocess
import sys


@contextlib.contextmanager
def suppress_stdout(suppress: bool = True) -> None:
    with open(os.devnull, mode="w") as devnull:
        stdout = sys.stdout
        if suppress:
            sys.stdout = devnull
        try:
            yield
        finally:
            if suppress:
                sys.stdout = stdout


def copy_to_clipboard(content: str) -> None:
    subprocess.run("pbcopy", universal_newlines=True, input=content, check=True)
