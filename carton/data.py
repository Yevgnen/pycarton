# -*- coding: utf-8 -*-

import statistics
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


def describe_series(
    s: Sequence, r: int = 2, qs: Optional[Sequence[int]] = None
) -> Dict:
    if not qs:
        qs = [10, 25, 50, 75, 90, 95, 99]
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
    data: Sequence,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[Sequence] = None,
) -> Union[Dict[str, Any], Dict[str, Tuple]]:
    # pylint: disable=import-outside-toplevel
    from sklearn.model_selection import train_test_split

    def _new(x, data=data):
        return type(data)(x)

    def _split(data, size, stratify=None):
        params = (data, stratify) if stratify is not None else (data,)
        splits = train_test_split(
            *params,
            test_size=size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        return splits

    val_size /= 1 - test_size
    splits = _split(data, test_size, stratify=stratify)

    if stratify is None:
        train_val, test = splits
        train, val = _split(train_val, val_size, stratify=stratify)
        return {
            "train": _new(train),
            "val": _new(val),
            "test": _new(test),
        }

    train_val_x, test_x, train_val_y, test_y = splits
    train_x, val_x, train_y, val_y = _split(train_val_x, val_size, stratify=train_val_y)

    return {
        "train": (_new(train_x), _new(train_y, stratify)),
        "val": (_new(val_x), _new(val_y, stratify)),
        "test": (_new(test_x), _new(test_y, stratify)),
    }
