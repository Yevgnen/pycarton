# -*- coding: utf-8 -*-

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

import numpy as np


def describe_series(
    s: Sequence, r: int = 2, qs: Optional[Sequence[int]] = None
) -> dict:
    if not qs:
        qs = [10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
    info = {
        "size": len(s),
        "mode": statistics.mode(s),
        "mean": float(np.mean(s).round(r)),
        "std": float(np.std(s).round(r)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
    }

    percentiles = np.percentile(s, qs).round(r)
    for q, p in zip(qs, percentiles):
        info.update({f"{q}%": p.round(r)})

    return info


def split(
    *data: Iterable[Sequence],
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[Sequence] = None,
) -> Union[dict[str, Any], dict[str, tuple]]:
    # pylint: disable=import-outside-toplevel
    from sklearn.model_selection import train_test_split

    def _split(data, size, stratify=None):
        params = data + (stratify,) if stratify is not None else data
        splits = train_test_split(
            *params,
            test_size=size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        split0, split1 = tuple(splits[::2]), tuple(splits[1::2])

        return split0, split1

    def _format_output(x):
        if stratify:
            x = x[:-1]

        return x if len(x) > 1 else x[0]

    val_size /= 1 - test_size
    train_val, test = _split(data, test_size, stratify=stratify)

    if stratify is None:
        train, val = _split(train_val, val_size, stratify=None)
    else:
        *train_val, train_val_stratify = train_val
        train, val = _split(tuple(train_val), val_size, stratify=train_val_stratify)

    return {
        "train": _format_output(train),
        "val": _format_output(val),
        "test": _format_output(test),
    }
