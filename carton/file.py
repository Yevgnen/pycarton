# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import itertools
import math
import multiprocessing
import os
from collections.abc import Callable, Generator, Iterable
from typing import Any, Optional, Union


def normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


class _Path(str):
    def __new__(cls, value):
        return super().__new__(cls, (value))

    def __truediv__(self, path):
        return _Path(os.path.join(self, path))

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
        sorted_group = sorted(group, key=lambda x: os.path.splitext(x)[1])
        if len(sorted_group) != num_exts and missing == "error":
            raise RuntimeError(f"Missing files: {key}.{exts}")

        yield (key, sorted_group) if with_key else sorted_group


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


def iter_lines(
    filename: str,
    ignore_empty: bool = True,
    strip: bool = True,
    transform: Callable[[str], str] = lambda x: x,
) -> Iterable[str]:
    with open(filename, mode="r") as f:
        for line in f:
            if strip:
                line = line.rstrip()

            if line or not ignore_empty:
                yield transform(line)


def _get_chunk_size(filename, num_workers=multiprocessing.cpu_count()):
    return math.ceil(os.path.getsize(filename) / num_workers)


def _get_chunkified_args(
    filename, *args, num_workers=multiprocessing.cpu_count(), chunk_size=None
):
    if not chunk_size:
        chunk_size = _get_chunk_size(filename, num_workers=num_workers)

    return [
        (filename, *args, start, end)
        for filename, *args, (start, end) in zip(
            itertools.repeat(filename),
            *[itertools.repeat(arg) for arg in args],
            chunkify(filename, chunk_size=chunk_size),
        )
    ]


def _iter_line_wrapper(filename, fn, start, end):
    return [fn(line) for line in readlines(filename, start, end)]


def chunkify(
    filename: Union[str, os.PathLike], chunk_size: int = 1024 * 1024
) -> Generator[tuple[int, int], None, None]:
    start = 0
    size = os.path.getsize(filename)
    with open(filename, mode="rb") as f:
        while True:
            f.seek(chunk_size, 1)
            f.readline()
            end = min(f.tell(), size)
            yield start, end

            if end >= size:
                break
            start = end


def readlines(
    filename: Union[str, os.PathLike],
    start: Optional[int] = None,
    end: Optional[int] = None,
    **kwargs,
) -> Generator[str, None, None]:
    with open(filename, **kwargs) as f:
        if start:
            f.seek(start)
        while True:
            line = f.readline()
            if end and f.tell() > end or not line:
                break

            yield line


def map_text(
    filename: Union[str, os.PathLike],
    fn: Callable[[str, int, int], Any],
    num_workers: int = multiprocessing.cpu_count(),
    chunk_size: Optional[int] = None,
) -> Iterable:
    args = _get_chunkified_args(
        filename, num_workers=num_workers, chunk_size=chunk_size
    )
    with multiprocessing.Pool(processes=num_workers) as p:
        data = p.starmap(fn, args)

    return itertools.chain.from_iterable(data)


def map_lines(
    filename: Union[str, os.PathLike],
    fn: Callable[[str], Any],
    num_workers: int = multiprocessing.cpu_count(),
    chunk_size: Optional[int] = None,
) -> Iterable:
    args = _get_chunkified_args(
        filename, fn, num_workers=num_workers, chunk_size=chunk_size
    )
    with multiprocessing.Pool(processes=num_workers) as p:
        data = p.starmap(_iter_line_wrapper, args)

    return itertools.chain.from_iterable(data)
