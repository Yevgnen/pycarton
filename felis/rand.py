# -*- coding: utf-8 -*-

import random
from typing import List, Optional, Union

from felis.palette import Colors


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
