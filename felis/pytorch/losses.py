# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, positive_logits, negative_logits):
        return torch.mean(
            torch.clamp(negative_logits + self.margin - positive_logits, 0)
        )
