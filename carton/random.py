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
    """Generate random seed.

    Args:
        seed: If not None, this function does nothing other than return
            the seed. (default: None)
        low: Lower bound (inclusive) of the generation. (default: 0)
        high: Higher bound (inclusive) of the generation. (default: 4294967295)

    Returns:
        seed: The generated random seed.
    """
    if seed is not None:
        return seed

    return random.randint(low, high)


def set_seed(seed: int, debug: bool = False) -> None:
    """Set random states of some modules.

    This function will set the random states of follwing libraries:

    - :mod:`random` by
      - :func:`random.seed`
    - :mod:`numpy` (if found) by
      - :func:`numpy.random.seed`
    - :mod:`torch` (if found) by
      - :func:`torch.manual_seed`
      - :func:`torch.random.manual_seed`
      - :func:`torch.cuda.manual_seed`
      - :func:`torch.cuda.manual_seed_all`

    Args:
        seed: Random state to be set.
        debug: Also set debug state if True. (default: false)

    Returns:
        None
    """

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
    """Generate random strings.

    Args:
        length: Length of the string to be generated. (default: 10)
        chars: Chars to be used to generate the string.
            (default: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    Returns:
        string: The generated random string in given length.
    """

    return "".join(random.choices(chars, k=length))


def random_colors(
    colors: Optional[list[str]] = None,
    num: int = 1,
    replacement: bool = False,
) -> Union[str, list[str]]:
    """Sample random colors.

    Args:
        colors: Colors to be sampled. If None, preset colors in
            :attr:`Colors.PRESETS` will be used. (default: None)
        num: Number of random colors to sample.
        replacement: Sample with replacement if True. (default: False)

    Returns:
        colors: List of sampled colors or single sampled color.
    """
    if colors is None:
        colors = Colors.PRESETS

    sample_function = random.choices if replacement else random.sample
    selected = sample_function(colors, k=num)

    return selected if num > 1 else selected[0]
