# -*- coding: utf-8 -*-

import collections
import warnings
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

from carton.utils import identity


def iterable(x: Any) -> bool:
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))


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
    collate_fn: Union[Callable, Mapping[str, Callable]] = identity,
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
