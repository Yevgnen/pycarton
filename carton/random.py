# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import importlib
import random
from collections.abc import Iterator
from typing import Optional, Union

from carton.palette import Colors


def random_state(
    seed: Optional[int] = None, low: int = 0, high: int = 2 ** 32 - 1
) -> int:
    if seed is not None:
        return seed

    return random.randint(low, high)


def _import_module(module):
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError:
        return None

    return module


def set_seed(seed: int, debug: bool = False) -> tuple:
    states: tuple = (random.getstate(),)

    random.seed(seed)

    numpy = _import_module("numpy")
    if numpy:
        states += (numpy.random.get_state(),)
        numpy.random.seed(seed)

    torch = _import_module("torch")
    if torch:
        cuda = torch.cuda.is_available()

        states += (
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if cuda else None,
            torch.cuda.get_rng_state_all() if cuda else None,
        )
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if debug:
            states += (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
            )
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return states


@contextlib.contextmanager
def seed(random_seed: int) -> Iterator[None]:
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
