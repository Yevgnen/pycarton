# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch


def length_to_mask(
    lengths: torch.Tensor,
    max_len: Optional[Union[int, torch.Tensor]] = None,
    batch_first: bool = False,
) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.max()

    mask = torch.arange(max_len, device=lengths.device).unsqueeze(dim=1).expand(
        max_len, len(lengths)
    ) < lengths.unsqueeze(dim=0)

    return mask.transpose(0, 1) if batch_first else mask


def mask_to_length(masks: torch.Tensor, batch_first: bool = False) -> torch.Tensor:
    return masks.sum(dim=int(batch_first)).long()
