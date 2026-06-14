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

"""Offline PI0 vision-feature store for AFT.

Loads the pre-extracted PI0 vision-tower features (the offline features documented
in ``lerobot/pi0_feature_extraction_flow.md`` /
``pi0_feature_extraction_flow_update.md``), pools them into a single fixed-size
vector per frame, and exposes an ``(episode_index, frame_index)`` lookup so the
Diffusion Policy training loop can fetch the matching pre-trained visual feature
for each sampled frame.

Two on-disk formats are supported (auto-detected by file extension):

1. ``*.pt`` — the actual extractor output (``torch.save`` dict). Layout::

       {
         'args': {... extraction config ...},
         'rows': [                                  # one dict per frame
            {
              'episode_index': int,
              'frame_index'  : int,
              'task_index'   : int,
              ...,
              'internal_features': {
                 'vision_tower'  : Tensor (num_camera_calls, batch=1, image_tokens, 1152),
                 'paligemma_last': Tensor (1, 1, prefix_tokens, 2048),  # unused here
              },
            },
            ...
         ],
       }

   A confirmed ``pi0_libero_base`` shard (episode 1, 336 frames) had per-frame
   ``internal_features['vision_tower']`` of shape ``(3, 1, 256, 1152)`` in
   ``bfloat16``.

2. ``*.safetensors`` — a flattened/stacked layout (kept for forward compatibility)::

       episode_index         : (num_frames,)
       frame_index           : (num_frames,)
       features.vision_tower  : (num_frames, num_camera_calls, batch, image_tokens, 1152)

Camera slots: PI0 emits ``num_camera_calls`` slots (3 in the observed data): slot
0 = base (LIBERO ``observation.images.image``), slot 1 = left wrist (LIBERO
``observation.images.wrist_image``), slot 2 = a **dummy missing right-wrist
camera** that must be excluded. Use ``camera_indices=(0, 1)`` to keep only the two
real cameras.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

try:
    from safetensors import safe_open as _safe_open
except Exception:  # pragma: no cover - safetensors is a hard dep of lerobot, but stay defensive
    _safe_open = None

# Keys under which the per-frame vision-tower feature may live (in priority order).
_VISION_KEYS = ("vision_tower", "features.vision_tower")
_INTERNAL_KEYS = ("internal_features", "features")


def _to_int(x) -> int:
    """Coerce a python int / numpy scalar / 0-d or 1-element tensor to a python int."""
    if isinstance(x, torch.Tensor):
        return int(x.reshape(-1)[0].item())
    return int(x)


def _find_vision(container: dict) -> torch.Tensor | None:
    """Locate the vision-tower tensor inside a row dict or a flat shard dict."""
    # Direct keys (e.g. flat safetensors 'features.vision_tower', or a bare 'vision_tower').
    for k in _VISION_KEYS:
        v = container.get(k)
        if isinstance(v, torch.Tensor):
            return v
    # Nested under 'internal_features' / 'features'.
    for outer in _INTERNAL_KEYS:
        sub = container.get(outer)
        if isinstance(sub, dict):
            for k in _VISION_KEYS:
                v = sub.get(k)
                if isinstance(v, torch.Tensor):
                    return v
    return None


def _select_cameras(per_frame: torch.Tensor, camera_indices: tuple[int, ...] | None) -> torch.Tensor:
    """Keep only the requested camera slots along the camera axis (dim 0) of a per-frame feature.

    A per-frame vision feature is ``(num_cameras, [batch], image_tokens, dim)``; the
    camera axis is the leading dimension. A 2-D ``(image_tokens, dim)`` single-camera
    feature has no camera axis and is returned unchanged.
    """
    if camera_indices is None or per_frame.dim() < 3:
        return per_frame
    n_cam = per_frame.shape[0]
    valid = [i for i in camera_indices if 0 <= i < n_cam]
    dropped = [i for i in camera_indices if i not in valid]
    if dropped:
        logging.warning(
            "PI0VisionFeatureStore: camera_indices %s out of range for %d camera slots; using %s.",
            dropped,
            n_cam,
            valid,
        )
    if not valid:
        return per_frame
    return per_frame[list(valid)]


def _pool_tokens(per_frame: torch.Tensor, token_pool: str, camera_indices: tuple[int, ...] | None) -> torch.Tensor:
    """Pool a single frame's vision-tower feature into a ``(num_cameras, D)`` tensor.

    Accepts the per-frame slice ``(num_camera_calls, batch, image_tokens, D)`` (and a
    few degenerate variants), selects the real camera slots, then pools over the
    image-token axis.
    """
    t = per_frame.float()
    t = _select_cameras(t, camera_indices)
    # Collapse a singleton batch dim if present: (C, B, T, D) -> (C, T, D).
    if t.dim() == 4 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.dim() == 4:
        # (C, B, T, D) with B > 1: average over batch too.
        t = t.mean(dim=1)
    if t.dim() == 2:
        # (T, D) single camera -> add a camera axis.
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(
            f"Unexpected per-frame vision feature shape {tuple(per_frame.shape)}; "
            "expected (num_cameras, [batch], image_tokens, dim)."
        )
    # t: (num_cameras, image_tokens, D) -> pool over tokens.
    if token_pool == "mean":
        return t.mean(dim=1)
    if token_pool == "max":
        return t.max(dim=1).values
    if token_pool == "cls":
        return t[:, 0]
    raise ValueError(f"Unknown token_pool {token_pool!r}; expected 'mean', 'max', or 'cls'.")


def _reduce_cameras(cam_feats: torch.Tensor, camera_reduce: str) -> torch.Tensor:
    """Reduce a (num_cameras, D) tensor to a single per-frame vector."""
    if camera_reduce == "concat":
        return cam_feats.reshape(-1)
    if camera_reduce == "mean":
        return cam_feats.mean(dim=0)
    raise ValueError(f"Unknown camera_reduce {camera_reduce!r}; expected 'concat' or 'mean'.")


class PI0VisionFeatureStore:
    """In-memory lookup of pooled PI0 vision features keyed by (episode, frame).

    Args:
        feature_dir: Directory holding the ``*.pt`` and/or ``*.safetensors`` shards.
        token_pool: How to pool over the image-token axis (``"mean"``/``"max"``/``"cls"``).
        camera_reduce: How to combine the per-camera vectors (``"concat"``/``"mean"``).
        camera_indices: Which camera slots to keep before pooling. ``None`` keeps all
            slots; pass e.g. ``(0, 1)`` to drop the dummy right-wrist slot (index 2).
            Out-of-range indices are filtered with a warning.
        globs: Glob pattern(s) for shard files inside ``feature_dir``.
    """

    def __init__(
        self,
        feature_dir: str | Path,
        token_pool: str = "mean",
        camera_reduce: str = "concat",
        camera_indices: tuple[int, ...] | None = None,
        globs: str | tuple[str, ...] = ("*.pt", "*.safetensors"),
    ):
        self.feature_dir = Path(feature_dir)
        self.token_pool = token_pool
        self.camera_reduce = camera_reduce
        self.camera_indices = tuple(camera_indices) if camera_indices is not None else None
        self.globs = (globs,) if isinstance(globs, str) else tuple(globs)
        self._index: dict[tuple[int, int], int] = {}
        self._features: torch.Tensor | None = None
        self.feature_dim: int = 0

        self._build()

    @property
    def is_empty(self) -> bool:
        return self._features is None or self._features.numel() == 0

    def _discover_shards(self) -> list[Path]:
        if not self.feature_dir.exists():
            return []
        shards: list[Path] = []
        for pattern in self.globs:
            shards.extend(self.feature_dir.glob(pattern))
        # Deduplicate while keeping a stable, reproducible order.
        return sorted(set(shards))

    def _iter_records(self, shard: Path):
        """Yield ``(episode_index, frame_index, per_frame_vision_tensor)`` from one shard."""
        if shard.suffix == ".pt":
            # Trusted local extractor output; contains python ints/strings/lists so
            # weights_only=False is required.
            obj = torch.load(str(shard), map_location="cpu", weights_only=False)
            rows = obj["rows"] if isinstance(obj, dict) and "rows" in obj else obj
            if not isinstance(rows, (list, tuple)):
                logging.warning("PI0VisionFeatureStore: %s has no 'rows' list, skipping.", shard)
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                vis = _find_vision(row)
                if vis is None or "episode_index" not in row or "frame_index" not in row:
                    continue
                yield _to_int(row["episode_index"]), _to_int(row["frame_index"]), vis
        else:  # .safetensors (flat, stacked)
            if _safe_open is None:
                logging.warning("PI0VisionFeatureStore: safetensors not importable; skipping %s.", shard)
                return
            # Read ONLY the keys we need (vision_tower + indices). This avoids
            # materializing sibling tensors such as the large `features.paligemma_last`.
            with _safe_open(str(shard), framework="pt") as sf:
                keys = set(sf.keys())
                vkey = next((k for k in _VISION_KEYS if k in keys), None)
                if vkey is None or "episode_index" not in keys or "frame_index" not in keys:
                    logging.warning("PI0VisionFeatureStore: vision/index keys missing in %s, skipping.", shard)
                    return
                ep = sf.get_tensor("episode_index").to(torch.int64).reshape(-1)
                fr = sf.get_tensor("frame_index").to(torch.int64).reshape(-1)
                vis = sf.get_tensor(vkey)
            for f in range(vis.shape[0]):
                yield int(ep[f]), int(fr[f]), vis[f]

    def _build(self) -> None:
        shards = self._discover_shards()
        if not shards:
            logging.warning(
                "PI0VisionFeatureStore: no shards matching %s in '%s'. "
                "AFT vision regularization will be inactive (training falls back to plain Diffusion Policy).",
                list(self.globs),
                self.feature_dir,
            )
            return

        pooled_rows: list[torch.Tensor] = []
        keys: list[tuple[int, int]] = []
        for shard in shards:
            for ep, fr, vis in self._iter_records(shard):
                pooled = _reduce_cameras(
                    _pool_tokens(vis, self.token_pool, self.camera_indices), self.camera_reduce
                )
                pooled_rows.append(pooled)
                keys.append((ep, fr))

        if not pooled_rows:
            logging.warning("PI0VisionFeatureStore: shards found but no usable features in %s.", self.feature_dir)
            return

        self._features = torch.stack(pooled_rows, dim=0).contiguous()  # (N, D) on CPU
        self.feature_dim = self._features.shape[1]
        for i, key in enumerate(keys):
            # Last writer wins on duplicate (episode, frame) keys.
            self._index[key] = i
        logging.info(
            "PI0VisionFeatureStore: loaded %d frames from %d shard(s), pooled dim=%d, token_pool=%s, "
            "camera_reduce=%s, camera_indices=%s, from %s",
            self._features.shape[0],
            len(shards),
            self.feature_dim,
            self.token_pool,
            self.camera_reduce,
            self.camera_indices,
            self.feature_dir,
        )

    @torch.no_grad()
    def lookup(self, episode_index: torch.Tensor, frame_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch pooled pre-trained features for a batch of (episode, frame) ids.

        Args:
            episode_index: ``(B,)`` (or broadcastable) integer episode ids.
            frame_index: ``(B,)`` (or broadcastable) integer frame ids.

        Returns:
            ``(features, mask)`` where ``features`` is ``(B, feature_dim)`` (zeros
            for frames not present in the store) and ``mask`` is a ``(B,)`` bool
            tensor that is True for frames that were found. Returned tensors live
            on CPU; the caller moves them to the model device.
        """
        ep = episode_index.detach().to("cpu").reshape(episode_index.shape[0], -1)[:, -1].to(torch.int64)
        fr = frame_index.detach().to("cpu").reshape(frame_index.shape[0], -1)[:, -1].to(torch.int64)
        batch = ep.shape[0]
        out = torch.zeros(batch, self.feature_dim, dtype=torch.float32)
        mask = torch.zeros(batch, dtype=torch.bool)
        if self.is_empty:
            return out, mask
        for i in range(batch):
            row = self._index.get((int(ep[i]), int(fr[i])))
            if row is not None:
                out[i] = self._features[row]
                mask[i] = True
        return out, mask
