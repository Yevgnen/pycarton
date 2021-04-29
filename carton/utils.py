# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from typing import TypeVar

T = TypeVar("T")


def better_display() -> None:
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.unicode.ambiguous_as_wide", True)


def identity(x: T) -> T:
    return x


def git_version(dirname: str) -> str:
    if not shutil.which("git"):
        return None

    dirname = os.path.abspath(os.path.expanduser(dirname))
    if not os.path.isdir(dirname):
        dirname = os.path.dirname(dirname)

    cmd = [
        "git",
        "--no-pager",
        "log",
        "--pretty=oneline",
        "--abbrev-commit",
        "-1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=dirname,
        )
    except subprocess.CalledProcessError:
        return None

    return proc.stdout.decode().strip()
