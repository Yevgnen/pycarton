# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import itertools
import os
from collections.abc import Iterable
from typing import Union


def normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


class _Path(str):
    def __new__(cls, value):
        return super().__new__(cls, (value))

    def __call__(self, *paths):
        paths = (self,) + tuple(paths)

        return _Path(os.path.join(*paths))

    def norm(self):
        return self.__class__(normalize_path(self))


def path(root: str) -> _Path:
    return _Path(root)


def iter_file_groups(
    dirname: str,
    exts: Union[str, Iterable[str]],
    with_key: bool = False,
    missing: str = "error",
) -> Union[
    Iterable[str], Iterable[tuple[str, ...]], tuple[str, Iterable[tuple[str, ...]]]
]:
    def _format_ext(ext):
        return f'.{ext.lstrip(".")}'

    def _iter_files(dirname):
        for dirpath, _, filenames in os.walk(dirname):
            for filename in filenames:
                if os.path.splitext(filename)[1] in exts:
                    yield os.path.join(dirpath, filename)

    missings = {"error", "ignore"}
    if missing not in missings:
        raise ValueError(f"Param `missing` should be in {missings}")

    if isinstance(exts, str):
        return glob.iglob(
            os.path.join(dirname, f"**/*{_format_ext(exts)}"), recursive=True
        )

    exts = {*map(_format_ext, exts)}
    num_exts = len(exts)
    files = _iter_files(dirname)
    for key, group in itertools.groupby(
        sorted(files), key=lambda x: os.path.splitext(os.path.relpath(x, dirname))[0]
    ):
        group = sorted(group, key=lambda x: os.path.splitext(x)[1])
        if len(group) != num_exts and missing == "error":
            raise RuntimeError(f"Missing files: {key}.{exts}")

        yield (key, group) if with_key else group


def read_lines(filename: str, chunk_size: int = 64 * 1024) -> Iterable[str]:
    with open(filename, mode="r") as f:
        remaining = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            linefeed = chunk.rfind("\n")
            if linefeed > 0:
                chunk, remaining = remaining + chunk[:linefeed], chunk[linefeed + 1 :]
            else:
                remaining = ""

            for line in chunk.split("\n"):
                yield line
