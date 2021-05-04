# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import importlib
import random
from typing import Optional, Union

from carton.palette import Colors


def random_state(
    seed: Optional[int] = None, low: int = 0, high: int = 2 ** 32 - 1
) -> int:
    if seed is not None:
        return seed

    return random.randint(low, high)


def set_seed(seed: int, debug: bool = False) -> None:
    def _import(module):
        try:
            module = importlib.import_module(module)
        except ModuleNotFoundError:
            return None

        return module

    random.seed(seed)

    numpy = _import("numpy")
    if numpy:
        numpy.random.seed(seed)

    torch = _import("torch")
    if torch:
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if debug:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


@contextlib.contextmanager
def seed(random_seed: int):
    state = random.getstate()
    random.seed(random_seed)
    try:
        yield
    finally:
        random.setstate(state)


def random_string(
    length: int = 10,
    chars: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
) -> str:
    return "".join(random.choices(chars, k=length))


def random_colors(
    colors: Optional[list[str]] = None,
    num: int = 1,
    replacement: bool = False,
) -> Union[str, list[str]]:
    if colors is None:
        colors = Colors.PRESETS

    sample_function = random.choices if replacement else random.sample
    selected = sample_function(colors, k=num)

    return selected if num > 1 else selected[0]
