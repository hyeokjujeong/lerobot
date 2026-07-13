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
import re
from collections import OrderedDict
from pathlib import Path

import torch

try:
    from safetensors import safe_open as _safe_open
except Exception:  # pragma: no cover - safetensors is a hard dep of lerobot, but stay defensive
    _safe_open = None

# Keys under which the default per-frame vision-tower feature may live (in priority order).
_VISION_KEYS = ("vision_tower", "features.vision_tower")
_INTERNAL_KEYS = ("internal_features", "features")
_EPISODE_RE = re.compile(r"episode_(\d+)")


def _to_int(x) -> int:
    """Coerce a python int / numpy scalar / 0-d or 1-element tensor to a python int."""
    if isinstance(x, torch.Tensor):
        return int(x.reshape(-1)[0].item())
    return int(x)


def _feature_key_candidates(feature_key: str) -> tuple[str, ...]:
    if feature_key.startswith("features."):
        bare = feature_key.removeprefix("features.")
        return (feature_key, bare)
    return (feature_key, f"features.{feature_key}")


def _find_vision(container: dict, feature_key: str = "features.vision_tower") -> torch.Tensor | None:
    """Locate the requested feature tensor inside a row dict or a flat shard dict."""
    # Direct keys (e.g. flat safetensors 'features.vision_tower', or a bare 'vision_tower').
    for k in _feature_key_candidates(feature_key):
        v = container.get(k)
        if isinstance(v, torch.Tensor):
            return v
    # Nested under 'internal_features' / 'features'.
    for outer in _INTERNAL_KEYS:
        sub = container.get(outer)
        if isinstance(sub, dict):
            for k in _feature_key_candidates(feature_key):
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


def _pool_cosmos3_g_v_kv0(per_frame: torch.Tensor, token_pool: str) -> torch.Tensor:
    """Pool Cosmos3 generator layer-0 visual K/V into a compact per-frame vector.

    Expected per-frame layout is either ``(kv=2, visual_tokens, kv_heads,
    head_dim)`` for a single/mosaic camera image or ``(num_cameras, kv=2,
    visual_tokens, kv_heads, head_dim)`` for separate-camera extraction. We use
    the same token-pooling idea as the R-ViT feature, then preserve K/V and head
    structure by flattening the remainder:

        (2, 128, 8, 128) -> mean over tokens -> (2, 8, 128) -> 2048
    """
    t = per_frame.float()
    if t.dim() == 4 and t.shape[0] == 2:
        t = t.unsqueeze(0)
    if t.dim() != 5 or t.shape[1] != 2:
        raise ValueError(
            f"Unexpected Cosmos3 G-V-KV0 feature shape {tuple(per_frame.shape)}; "
            "expected (2, visual_tokens, kv_heads, head_dim) or "
            "(num_cameras, 2, visual_tokens, kv_heads, head_dim)."
        )
    if token_pool == "mean":
        pooled = t.mean(dim=2)
    elif token_pool == "max":
        pooled = t.max(dim=2).values
    elif token_pool == "cls":
        pooled = t[:, :, 0]
    else:
        raise ValueError(f"Unknown token_pool {token_pool!r}; expected 'mean', 'max', or 'cls'.")
    return pooled.reshape(pooled.shape[0], -1)


def _pool_pretrained_feature(
    per_frame: torch.Tensor,
    feature_key: str,
    token_pool: str,
    camera_indices: tuple[int, ...] | None,
) -> torch.Tensor:
    if feature_key in ("cosmos3_g_v_kv0", "features.cosmos3_g_v_kv0"):
        return _pool_cosmos3_g_v_kv0(per_frame, token_pool)
    return _pool_tokens(per_frame, token_pool, camera_indices)


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
        feature_key: Tensor key to read from each shard. Defaults to the PI0/Cosmos
            R-ViT-compatible ``features.vision_tower``. Use
            ``features.cosmos3_g_v_kv0`` for Cosmos3 generator K/V features.
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
        feature_key: str = "features.vision_tower",
        token_pool: str = "mean",
        camera_reduce: str = "concat",
        camera_indices: tuple[int, ...] | None = None,
        globs: str | tuple[str, ...] = ("*.pt", "*.safetensors"),
    ):
        self.feature_dir = Path(feature_dir)
        self.feature_key = feature_key
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
                vis = _find_vision(row, self.feature_key)
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
                vkey = next((k for k in _feature_key_candidates(self.feature_key) if k in keys), None)
                if vkey is None or "episode_index" not in keys or "frame_index" not in keys:
                    logging.warning(
                        "PI0VisionFeatureStore: feature key %r or index keys missing in %s, skipping.",
                        self.feature_key,
                        shard,
                    )
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
                    _pool_pretrained_feature(vis, self.feature_key, self.token_pool, self.camera_indices),
                    self.camera_reduce,
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
            "PI0VisionFeatureStore: loaded %d frames from %d shard(s), feature_key=%s, pooled dim=%d, "
            "token_pool=%s, camera_reduce=%s, camera_indices=%s, from %s",
            self._features.shape[0],
            len(shards),
            self.feature_key,
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


class PI0ActionFeatureStore:
    """Lookup PI0 action-expert features keyed by (episode, frame, timestep).

    The expected extractor output is one or more ``*.safetensors`` shards with:

    - ``episode_index``: ``(N,)``
    - ``frame_index``: ``(N,)``
    - ``diffusion_timestep``: ``(N,)``
    - ``seed_repeat``: ``(N,)`` (optional for lookup; used only as row diversity)
    - ``features.<layer>``: ``(N, horizon, dim)`` or already-pooled ``(N, dim)``

    Multiple rows may exist for the same ``(episode, frame, timestep)`` because
    features can be extracted with several noise seeds. ``lookup`` randomly picks
    one matching row per sample; DP action noise is sampled independently by the
    caller.
    """

    def __init__(
        self,
        feature_dir: str | Path,
        layer: str = "pi0.layer17",
        timestep: int = 95,
        pool: str = "mean",
        store_mode: str = "lazy",
        max_cached_episodes: int = 4,
        globs: str | tuple[str, ...] = ("*.safetensors",),
    ):
        if _safe_open is None:
            raise RuntimeError("safetensors is required for PI0ActionFeatureStore.")
        if pool not in ("mean", "flatten"):
            raise ValueError(f"Unknown action feature pool {pool!r}; expected 'mean' or 'flatten'.")
        if store_mode not in ("lazy", "eager"):
            raise ValueError(f"Unknown store_mode {store_mode!r}; expected 'lazy' or 'eager'.")

        self.feature_dir = Path(feature_dir)
        self.layer = layer
        self.feature_key = f"features.{layer}"
        self.timestep = int(timestep)
        self.pool = pool
        self.store_mode = store_mode
        self.max_cached_episodes = int(max_cached_episodes)
        self.globs = (globs,) if isinstance(globs, str) else tuple(globs)

        self._episode_shards: dict[int, Path] = {}
        self._fallback_shards: list[Path] = []
        self._cache: OrderedDict[int, tuple[torch.Tensor, dict[tuple[int, int, int], list[int]]]] = (
            OrderedDict()
        )
        self._features: torch.Tensor | None = None
        self._index: dict[tuple[int, int, int], list[int]] = {}
        self.feature_dim: int = 0
        self._empty = True

        self._build()

    @property
    def is_empty(self) -> bool:
        return self._empty

    def _discover_shards(self) -> list[Path]:
        if not self.feature_dir.exists():
            return []
        shards: list[Path] = []
        for pattern in self.globs:
            shards.extend(self.feature_dir.glob(pattern))
        return sorted(set(shards))

    def _build(self) -> None:
        shards = self._discover_shards()
        if not shards:
            logging.warning(
                "PI0ActionFeatureStore: no shards matching %s in '%s'. Action AFT will be inactive.",
                list(self.globs),
                self.feature_dir,
            )
            return

        usable: list[Path] = []
        for shard in shards:
            with _safe_open(str(shard), framework="pt") as sf:
                keys = set(sf.keys())
                if self.feature_key not in keys or "episode_index" not in keys or "frame_index" not in keys:
                    continue
                if "diffusion_timestep" not in keys:
                    continue
                shape = sf.get_slice(self.feature_key).get_shape()
                dim = self._pooled_dim(shape[1:])
                if dim <= 0:
                    continue
                self.feature_dim = self.feature_dim or dim
                usable.append(shard)
                match = _EPISODE_RE.search(shard.stem)
                if match:
                    self._episode_shards[int(match.group(1))] = shard
                else:
                    self._fallback_shards.append(shard)

        if not usable:
            logging.warning(
                "PI0ActionFeatureStore: no usable shards with key '%s' in %s. Action AFT will be inactive.",
                self.feature_key,
                self.feature_dir,
            )
            return

        self._empty = False
        if self.store_mode == "eager":
            rows: list[torch.Tensor] = []
            offset = 0
            for shard in usable:
                features, index = self._load_shard(shard)
                rows.append(features)
                for key, ids in index.items():
                    self._index.setdefault(key, []).extend([offset + i for i in ids])
                offset += features.shape[0]
            self._features = torch.cat(rows, dim=0).contiguous() if rows else None
            self._empty = self._features is None or self._features.numel() == 0
            if self._features is not None:
                self.feature_dim = self._features.shape[1]
            logging.info(
                "PI0ActionFeatureStore: eagerly loaded %d rows from %d shard(s), dim=%d, layer=%s, "
                "timestep=%d, pool=%s.",
                0 if self._features is None else self._features.shape[0],
                len(usable),
                self.feature_dim,
                self.layer,
                self.timestep,
                self.pool,
            )
        else:
            logging.info(
                "PI0ActionFeatureStore: indexed %d lazy shard(s), dim=%d, layer=%s, timestep=%d, pool=%s.",
                len(usable),
                self.feature_dim,
                self.layer,
                self.timestep,
                self.pool,
            )

    def _pooled_dim(self, feature_shape: list[int] | tuple[int, ...]) -> int:
        if len(feature_shape) == 1:
            return int(feature_shape[0])
        if len(feature_shape) == 2:
            if self.pool == "mean":
                return int(feature_shape[1])
            return int(feature_shape[0] * feature_shape[1])
        return 0

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        if features.dim() == 2:
            return features.contiguous()
        if features.dim() != 3:
            raise ValueError(
                f"Unexpected PI0 action feature shape {tuple(features.shape)}; expected (N,D) or (N,T,D)."
            )
        if self.pool == "mean":
            return features.mean(dim=1).contiguous()
        return features.flatten(start_dim=1).contiguous()

    def _load_shard(self, shard: Path) -> tuple[torch.Tensor, dict[tuple[int, int, int], list[int]]]:
        with _safe_open(str(shard), framework="pt") as sf:
            keys = set(sf.keys())
            if self.feature_key not in keys:
                raise KeyError(f"{shard} does not contain {self.feature_key!r}.")
            features = self._reduce_features(sf.get_tensor(self.feature_key))
            ep = sf.get_tensor("episode_index").to(torch.int64).reshape(-1)
            fr = sf.get_tensor("frame_index").to(torch.int64).reshape(-1)
            ts = sf.get_tensor("diffusion_timestep").to(torch.int64).reshape(-1)

        index: dict[tuple[int, int, int], list[int]] = {}
        for i in range(features.shape[0]):
            if int(ts[i]) != self.timestep:
                continue
            key = (int(ep[i]), int(fr[i]), int(ts[i]))
            index.setdefault(key, []).append(i)
        return features, index

    def _get_episode_data(
        self, episode_index: int
    ) -> tuple[torch.Tensor, dict[tuple[int, int, int], list[int]]] | None:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        shard = self._episode_shards.get(episode_index)
        candidates = [shard] if shard is not None else self._fallback_shards
        for candidate in candidates:
            if candidate is None:
                continue
            features, index = self._load_shard(candidate)
            if not index and shard is None:
                continue
            data = (features, index)
            self._cache[episode_index] = data
            self._cache.move_to_end(episode_index)
            while len(self._cache) > max(self.max_cached_episodes, 1):
                self._cache.popitem(last=False)
            return data
        return None

    @torch.no_grad()
    def lookup(self, episode_index: torch.Tensor, frame_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one randomly selected PI0 action feature per batch item.

        Returns CPU tensors ``(features, mask)`` with shape ``(B, feature_dim)``
        and ``(B,)``. The randomness is only over pre-extracted seed repeats.
        """
        ep = episode_index.detach().to("cpu").reshape(episode_index.shape[0], -1)[:, -1].to(torch.int64)
        fr = frame_index.detach().to("cpu").reshape(frame_index.shape[0], -1)[:, -1].to(torch.int64)
        batch = ep.shape[0]
        out = torch.zeros(batch, self.feature_dim, dtype=torch.float32)
        mask = torch.zeros(batch, dtype=torch.bool)
        if self.is_empty:
            return out, mask

        for i in range(batch):
            key = (int(ep[i]), int(fr[i]), self.timestep)
            if self.store_mode == "eager":
                ids = self._index.get(key)
                features = self._features
            else:
                data = self._get_episode_data(int(ep[i]))
                if data is None:
                    continue
                features, index = data
                ids = index.get(key)
            if not ids or features is None:
                continue
            chosen = ids[int(torch.randint(len(ids), (1,)).item())]
            out[i] = features[chosen]
            mask[i] = True
        return out, mask
