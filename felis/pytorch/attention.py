# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.scaler = torch.sqrt(torch.tensor(self.hidden_size).float())
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attns=False):
        # `query`, `key`, `value`
        # shape: batch_size x max_len x hidden_size
        assert query.dim() == 3
        assert key.dim() == 3
        assert value.dim() == 3

        if query.size()[-1] != key.size()[-1]:
            raise ValueError(
                "mismatch size (dim: -1): `query`: {} and `key`: {}".format(
                    list(query.size()), list(key.size())
                )
            )

        if query.size()[-1] != self.hidden_size:
            raise ValueError(
                (
                    "mismatch size (dim: -1): `query` or `key` [{}]"
                    " and `hidden_size` [{}]"
                ).format(query.size()[-1], self.hidden_size)
            )

        # Compute raw attention scores, i.e. A = QK^{T}.
        # shape: batch_size x max_len x max_len
        attns = torch.bmm(query, key.transpose(1, 2))
        attns /= self.scaler

        # Normalization.
        if mask is not None:
            if mask.size() != attns.size():
                raise ValueError(
                    "mismatch size: `attns`: {} and `mask`: {}".format(
                        list(attns.size()), list(mask.size())
                    )
                )
            attns.masked_fill_(mask, float("-inf"))

        if return_attns:
            return value, attns

        # Apply softmax and dropout.
        attns = F.softmax(attns, dim=-1)
        attns = self.dropout(attns)

        # Weight `value` by attentions.
        outputs = torch.bmm(attns, value)

        return outputs


class FeedForwardAttention(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation is None:
            activation = nn.LeakyReLU()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.W_query = nn.Linear(in_channels, out_channels)
        self.W_key = nn.Linear(in_channels, out_channels)
        self.W_value = nn.Linear(in_channels, out_channels)
        self.fc_query = nn.Linear(out_channels, 1)
        self.fc_key = nn.Linear(out_channels, 1)

    def forward(self, query, key, value, mask=None, return_attns=False):
        assert query.dim() == 3
        assert key.dim() == 3
        assert value.dim() == 3

        if len(set([query.size()[-1], key.size()[-1], value.size()[-1]])) != 1:
            raise ValueError(
                (
                    "mismatch size (dim: -1) of `query`: {} and `key`: {}"
                    " and `value`: {}"
                ).format(list(query.size()), list(key.size()), list(value.size()))
            )

        # `query`, `key`, `value` should have same hidden size.
        if query.size()[-1] != self.in_channels:
            raise ValueError(
                (
                    "mismatch size: dim -1 of `query` or `key` or `value` [{}]"
                    " and `in_channels` [{}]"
                ).format(query.size()[-1], self.in_channels)
            )

        # `query`, `key`, `value`
        # shape: batch_size x max_len x in_channels
        # `weighted_query`, `weighted_key`, `weighted_value`
        # shape: batch_size x max_len x out_channels
        w_key = self.W_key(key)
        w_query = self.W_query(query) if query is not key else w_key
        w_value = self.W_value(value) if value is not key else w_key

        # Feed to a single layer MLP.
        # `attns`
        # shape: batch_size x max_len x max_len
        fc_query = self.fc_query(w_query)
        fc_key = self.fc_key(w_key)
        attns = fc_query + fc_key.transpose(1, 2)

        # Normalization.
        if mask is not None:
            if mask.size() != attns.size():
                raise ValueError(
                    "mismatch size: `attns`: {} and `mask`: {}".format(
                        list(attns.size()), list(mask.size())
                    )
                )
            attns.masked_fill_(mask, float("-inf"))

        if return_attns:
            return w_value, attns

        # Apply softmax and dropout.
        attns = F.softmax(attns, dim=-1)
        attns = self.dropout(attns)

        # Weight `w_value` by attentions.
        outputs = torch.bmm(attns, w_value)

        # Apply activation.
        attns = self.activation(attns)

        return outputs


class BidirectionalAttention(nn.Module):
    def forward(self, v1, v1_mask, v2, v2_mask):
        similarity_matrix = v1.bmm(v2.transpose(2, 1))

        v2_v1_attn = F.softmax(
            similarity_matrix.masked_fill(v1_mask.unsqueeze(2), -float("inf")), dim=1
        )
        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(v2_mask.unsqueeze(1), -float("inf")), dim=2
        )

        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2
