# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
# From https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/models/orthogonal.py

import math
import torch

from torch import nn
from torch_householder import torch_householder_orgqr


class Orthogonal(nn.Module):
    """Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal"""
    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        if orthogonal_map == "householder":
            from torch_householder import torch_householder_orgqr
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params):
        assert params.size(-1) == 1
        sin = torch.sin(params)
        cos = torch.cos(params)
        return torch.stack([torch.cat([cos, sin], dim=1), torch.cat([-sin, cos], dim=1)], dim=2)

    def get_2d_orth(self, params):
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 2
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        sin0, sin1 = sin[:, :1], sin[:, 1:]
        cos0, cos1 = cos[:, :1], cos[:, 1:]
        rot = torch.stack([torch.cat([cos0, sin0], dim=1), torch.cat([-sin0, cos0], dim=1)], dim=2)
        orth = torch.stack([torch.cat([-cos1, sin1], dim=1), torch.cat([sin1, cos1], dim=1)], dim=2)
        # want them to alternate:
        rot = rot.reshape([-1, 1, 2, 2])
        orth = orth.reshape([-1, 1, 2, 2])
        return torch.cat([rot, orth], dim=1).reshape([-1, 2, 2])

    def get_3d_rotation(self, params):
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        r1 = torch.cat([cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g], dim=1)
        r2 = torch.cat([cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g], dim=1)
        r3 = torch.cat([-sin_b, cos_b*sin_g, cos_b*cos_g], dim=1)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params is [num_nodes*num_bundles, d(d-1)/2]
        if self.orthogonal_map == 'householder':
            B = params.size(0)
            nb = int(self.d*(self.d - 1) / 2)
            if self.d > 2:
                tril_ind = torch.tril_indices(self.d, self.d, offset=-1, device=params.device)  # for batch size =1 case https://stackoverflow.com/questions/74551428/select-pytorch-tensor-elements-by-list-of-indices
            else:
                tril_ind = torch.tensor([[1], [0]], device=params.device)  # for some reason the previous doesn't work for d=2
            A = torch.zeros([B, self.d, self.d], dtype=torch.float, device=params.device)
            A[:, tril_ind[0], tril_ind[1]] = params

            eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
            A = A + eye
            Q = torch_householder_orgqr(A)
        elif self.orthogonal_map == 'euler':
            assert 2 <= self.d <= 3
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        return Q
