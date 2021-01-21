# -*- coding: utf-8 -*-

import collections
import os
import random
import shutil
import subprocess
import warnings
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union


def better_display() -> None:
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.unicode.ambiguous_as_wide", True)


def identity(x: Any) -> Any:
    return x


def iterable(x: Any) -> bool:
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))


def random_state(
    seed: Optional[int] = None, low: int = 0, high: int = 2 ** 32 - 1
) -> int:
    if seed is not None:
        return seed

    return random.randint(low, high)


def set_seed(seed: int, debug: bool = False) -> None:
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if debug:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


class _Path(str):
    def __new__(cls, value):
        return super().__new__(cls, normalize_path(value))

    def __call__(self, *paths):
        paths = (self,) + paths
        paths = tuple(os.path.normpath(p) for p in paths)

        return _Path(os.path.join(*paths))


def path(root: str) -> _Path:
    return _Path(root)


def flatten_dict(d: Mapping, sep: str = ".") -> Dict[str, Any]:
    flattened = {}

    def _flatten(d, prefix):
        for key, value in d.items():
            key = str(key)
            if sep in key:
                raise ValueError(f"`sep` in `key` ({key!r}) is not allowed")

            new_prefix = sep.join([prefix, key]) if prefix else key
            if not isinstance(value, collections.abc.Mapping):
                flattened[new_prefix] = value
            else:
                _flatten(value, new_prefix)

    _flatten(d, "")

    return flattened


def roughen_dict(d: Mapping[str, Any], sep: str = ".") -> Dict[str, Any]:
    roughen = {}
    for key, value in d.items():
        if isinstance(value, collections.abc.Mapping):
            warnings.warn(f"Nested mapping found: {key!r}")

        if sep not in key:
            roughen[key] = value
            continue

        *keys, last_key = key.split(sep)
        child = roughen
        for subkey in keys:
            child = child.setdefault(subkey, {})
        child[last_key] = value

    return roughen


def chain_get(d: Mapping, keys: Union[str, Sequence[str]], sep=".") -> Any:
    if isinstance(keys, str):
        if sep not in keys:
            return d[keys]
        keys = keys.split(sep)

    for key in keys:
        d = d[key]

    return d


def collate(
    data: Iterable[Mapping],
    keys: Optional[Sequence] = None,
    collate_fn: Callable = identity,
) -> Dict:
    def _get_collate_fn(key=None):
        if isinstance(collate_fn, collections.abc.Mapping):
            return collate_fn.get(key, identity)

        return collate_fn

    if keys is not None and (
        not isinstance(keys, collections.abc.Sequence) or isinstance(keys, str)
    ):
        keys = [keys]

    collated = collections.defaultdict(list)
    first_keys = None
    for x in data:
        if not isinstance(x, collections.abc.Mapping):
            raise TypeError(f"Not a mapping: {x}")

        if first_keys is None:
            first_keys = x.keys()
        elif first_keys != x.keys():
            raise ValueError(f"Inconsistent keys: {first_keys} != {x.keys()}")

        for key in keys or first_keys:
            collated[key] += [x[key]]

    collated = {key: _get_collate_fn(key)(value) for key, value in collated.items()}

    if len(collated) > 1:
        return collated

    return next(iter(collated.values()))


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
