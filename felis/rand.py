# -*- coding: utf-8 -*-

import random
from typing import List, Optional, Union

from felis.palette import Colors


def random_state(
    seed: Optional[int] = None, low: int = 0, high: int = 2 ** 32 - 1
) -> int:
    if seed is not None:
        return seed

    return random.randint(low, high)


def set_seed(seed: int, debug: bool = False) -> None:
    # pylint: disable=import-outside-toplevel
    try:
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
    except ModuleNotFoundError:
        pass


def random_string(
    length: int = 10,
    chars: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
) -> str:
    return "".join(random.choices(chars, k=length))


def random_colors(
    colors: Optional[List[str]] = None,
    num: int = 1,
    replacement: bool = False,
) -> Union[str, List[str]]:
    if colors is None:
        colors = Colors.PRESETS

    sample_function = random.choices if replacement else random.sample
    selected = sample_function(colors, k=num)

    return selected if num > 1 else selected[0]
