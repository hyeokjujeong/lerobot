# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision-only Adaptive Feature Transfer (AFT) prior.

This is a faithful, self-contained port of the ``KernelPrior`` from the AFT
reference implementation (``adaptive-feature-transfer/prior.py``,
``method=aft`` -> ``prior_type='kernel'``), specialized to a single pre-trained
feature source (the PI0 vision tower) and kept independent of the rest of the
AFT repo so it can be dropped into LeRobot without extra dependencies.

AFT idea (vision branch only)
-----------------------------
Let ``z`` be the downstream model's per-sample visual feature (here: the
Diffusion Policy RGB-encoder output) and ``g`` the pre-trained PI0 visual
feature for the same frame. AFT does *not* force ``z`` to equal ``g``. Instead
it matches the **kernel (Gram) matrices** computed over a mini-batch, after
adaptively *scaling* the pre-trained features with learnable per-dimension gates
``sigmoid(s)``. Those gates let the downstream model keep the task-relevant
subset of the pre-trained features and ignore the rest (the "adaptive" in AFT).

    target = g * sigmoid(s)            # adaptive feature selection
    z, target -> center (per dim) -> L2-normalize (per row)
    K   = z @ z.T                      # (B, B) downstream kernel
    K_t = target @ target.T            # (B, B) pre-trained kernel
    L_aft = beta * RMSE(K - K_t)       # regularization added to the task loss

Because only ``B x B`` kernels are compared, the downstream and pre-trained
feature dimensions do **not** need to match.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionKernelPrior(nn.Module):
    """AFT kernel-matching prior for a single pre-trained vision feature source.

    Args:
        num_features: Dimensionality of the (pooled) pre-trained PI0 vision
            feature. This is the size of the learnable adaptive scale vector.
        learn_scales: If True, the per-dimension gates ``sigmoid(s)`` are
            trainable (the "adaptive" feature selection). If False they are
            frozen at ``sigmoid(0) = 0.5`` (i.e. a plain, non-adaptive kernel
            match, equivalent to AFT with ``learn_scales=False``).
        kernel: ``"linear"`` (cosine Gram) or ``"rbf"`` (Gaussian).
    """

    def __init__(self, num_features: int, learn_scales: bool = True, kernel: str = "linear"):
        super().__init__()
        if kernel not in ("linear", "rbf"):
            raise ValueError(f"Unknown kernel {kernel!r}; expected 'linear' or 'rbf'.")
        self.num_features = int(num_features)
        self.kernel = kernel
        # Diagonal adaptive scales, initialised at 0 -> sigmoid(0) = 0.5 (matches AFT).
        self.s = nn.Parameter(torch.zeros(self.num_features))
        if not learn_scales:
            self.s.requires_grad = False
        # Lightweight metrics for logging (populated on each forward).
        self.metrics: dict[str, float] = {}

    def scale_pretrained(self, pretrained_feat: torch.Tensor) -> torch.Tensor:
        """Apply the learnable diagonal gates ``sigmoid(s)`` to pre-trained features."""
        return pretrained_feat * self.s.sigmoid()

    @staticmethod
    def _center_and_normalize(feat: torch.Tensor) -> torch.Tensor:
        # Center per dimension across the batch, then L2-normalize per row.
        feat = feat - torch.mean(feat, dim=0, keepdim=True)
        feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-8)
        return feat

    def _kernel(self, feat: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return torch.matmul(feat, feat.t())
        # rbf: exp(-||x_i - x_j||^2)
        sq = ((feat[:, None, :] - feat[None, :, :]) ** 2).sum(dim=-1)
        return torch.exp(-sq)

    def kernel_rmse(self, feat: torch.Tensor, pretrained_feat: torch.Tensor) -> torch.Tensor:
        """Return RMSE between the downstream and (adaptively scaled) pre-trained kernels.

        Args:
            feat: ``(B, d_down)`` downstream visual features.
            pretrained_feat: ``(B, d_pre)`` pre-trained PI0 visual features.

        Returns:
            Scalar RMSE (lower is more aligned). This equals ``-log_prob`` in the
            original AFT code.
        """
        target_feat = self.scale_pretrained(pretrained_feat)
        feat = self._center_and_normalize(feat)
        target_feat = self._center_and_normalize(target_feat)

        k = self._kernel(feat)
        k_target = self._kernel(target_feat)

        dk = k - k_target
        rmse = (torch.mean(dk**2) + 1e-8) ** 0.5

        # Detached metrics for logging only.
        with torch.no_grad():
            self.metrics = {
                "aft_kernel_rmse": rmse.item(),
                "aft_scales_mean": self.s.sigmoid().mean().item(),
                "aft_scales_std": self.s.sigmoid().std().item(),
            }
        return rmse

    def loss(self, feat: torch.Tensor, pretrained_feat: torch.Tensor, beta: float) -> torch.Tensor:
        """AFT regularization term to *add* to the task loss: ``beta * kernel_rmse``."""
        return beta * self.kernel_rmse(feat, pretrained_feat)
