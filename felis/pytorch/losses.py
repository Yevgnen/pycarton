# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def manhattan_similarity(v1, v2):
    return torch.exp(-torch.norm(v1 - v2, p=1, dim=-1, keepdim=True))


class ManhattanSimilarity(nn.Module):
    def forward(self, v1, v2):
        return manhattan_similarity(v1, v2)


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, positive_logits, negative_logits):
        return torch.mean(
            torch.clamp(negative_logits + self.margin - positive_logits, 0)
        )
