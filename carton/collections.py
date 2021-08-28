# -*- coding: utf-8 -*-

from __future__ import annotations

import collections
import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Hashable, Optional, Union

from carton.utils import identity


def iterable(x: Any) -> bool:
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))


def dict_to_tuple(d: Mapping, keys: Optional[Iterable[Hashable]] = None) -> tuple:
    if not keys:
        return tuple(d.values())

    return tuple(d[x] for x in keys)


def get_dict_to_tuple_function(keys: Iterable[Hashable]) -> Callable[[Mapping], tuple]:
    keys = list(keys)

    def f(d):
        return dict_to_tuple(d, keys)

    return f


def flatten_dict(d: Mapping, sep: str = ".") -> dict[str, Any]:
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


def roughen_dict(d: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
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
    keys: Optional[Sequence[Hashable]] = None,
    collate_fn: Optional[Union[Callable, Mapping[Hashable, Callable]]] = None,
) -> dict:
    def _get_collate_fn(key=None):
        if isinstance(collate_fn, collections.abc.Mapping):
            return collate_fn.get(key, identity)

        return collate_fn

    it = iter(data)

    first: Mapping = next(it, {})
    first_keys = first.keys()

    if not keys:
        keys = list(first_keys)
    elif not iterable(keys):
        keys = [keys]

    collated: dict = {}
    for x in itertools.chain((first,), it):
        if not isinstance(x, collections.abc.Mapping):
            raise TypeError(f"Not a mapping: {x}")

        if first_keys != x.keys():
            raise ValueError(f"Inconsistent keys:{first_keys!r} != {x.keys()!r}")

        for key in keys:
            collated.setdefault(key, []).append(x[key])

    if collate_fn is not None:
        collated = {key: _get_collate_fn(key)(value) for key, value in collated.items()}

    if len(collated) > 1 or len(collated) == 0:
        return collated

    return next(iter(collated.values()))


def chunk(it: Iterable, size: int) -> Iterable:
    # Credit: https://stackoverflow.com/a/22045226/1831512
    it = iter(it)

    return iter(lambda: tuple(itertools.islice(it, size)), ())
