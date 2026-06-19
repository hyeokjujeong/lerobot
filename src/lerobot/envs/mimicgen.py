#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np

from .robosuite import ACTION_DIM, DEFAULT_CAMERAS, OBS_STATE_DIM, RobosuiteEnv
from .utils import _LazyAsyncVectorEnv, parse_camera_names

MIMICGEN_TASK_MODULES = {
    "Stack": "mimicgen.envs.robosuite.stack",
    "StackThree": "mimicgen.envs.robosuite.stack",
    "Square": "mimicgen.envs.robosuite.nut_assembly",
    "NutAssembly": "mimicgen.envs.robosuite.nut_assembly",
    "PickPlace": "mimicgen.envs.robosuite.pick_place",
    "Coffee": "mimicgen.envs.robosuite.coffee",
    "CoffeePreparation": "mimicgen.envs.robosuite.coffee",
    "MugCleanup": "mimicgen.envs.robosuite.mug_cleanup",
    "Threading": "mimicgen.envs.robosuite.threading",
    "ThreePieceAssembly": "mimicgen.envs.robosuite.three_piece_assembly",
    "HammerCleanup": "mimicgen.envs.robosuite.hammer_cleanup",
    "Kitchen": "mimicgen.envs.robosuite.kitchen",
}


def _mimicgen_module_for_task(task: str) -> str:
    for task_prefix, module_name in MIMICGEN_TASK_MODULES.items():
        if task == task_prefix or task.startswith(f"{task_prefix}_"):
            return module_name
    raise ValueError(
        f"Unknown MimicGen task '{task}'. Add its registration module to MIMICGEN_TASK_MODULES."
    )


class MimicGenEnv(RobosuiteEnv):
    """LeRobot gym.Env wrapper for MimicGen robosuite tasks."""

    def _register_task(self) -> None:
        module_name = _mimicgen_module_for_task(self.task)
        importlib.import_module(module_name)

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        observation = super()._format_raw_obs(raw_obs)
        pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            observation["pixels"] = {
                cam: np.flipud(image).copy() for cam, image in pixels.items()
            }
        return observation


def _make_env_fns(
    *,
    n_envs: int,
    task: str,
    robot: str,
    camera_names: list[str],
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    control_freq: int,
    episode_length: int,
    reward_shaping: bool,
    state_dim: int,
    action_dim: int,
) -> list[Callable[[], MimicGenEnv]]:
    def _make_env(episode_index: int) -> MimicGenEnv:
        return MimicGenEnv(
            task=task,
            robot=robot,
            camera_name=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            control_freq=control_freq,
            episode_length=episode_length,
            reward_shaping=reward_shaping,
            state_dim=state_dim,
            action_dim=action_dim,
            episode_index=episode_index,
        )

    return [partial(_make_env, i) for i in range(n_envs)]


def create_mimicgen_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = 400,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    robot = gym_kwargs.pop("robot", "Panda")
    obs_type = gym_kwargs.pop("obs_type", "pixels_agent_pos")
    render_mode = gym_kwargs.pop("render_mode", "rgb_array")
    observation_width = gym_kwargs.pop("observation_width", 256)
    observation_height = gym_kwargs.pop("observation_height", 256)
    control_freq = gym_kwargs.pop("control_freq", 10)
    reward_shaping = gym_kwargs.pop("reward_shaping", False)
    state_dim = gym_kwargs.pop("state_dim", OBS_STATE_DIM)
    action_dim = gym_kwargs.pop("action_dim", ACTION_DIM)
    camera_names = parse_camera_names(camera_name)

    if gym_kwargs:
        raise ValueError(f"Unexpected MimicGen kwargs: {sorted(gym_kwargs)}")

    fns = _make_env_fns(
        n_envs=n_envs,
        task=task,
        robot=robot,
        camera_names=camera_names,
        obs_type=obs_type,
        render_mode=render_mode,
        observation_width=observation_width,
        observation_height=observation_height,
        control_freq=control_freq,
        episode_length=episode_length,
        reward_shaping=reward_shaping,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    if env_cls is gym.vector.AsyncVectorEnv:
        out[task][0] = _LazyAsyncVectorEnv(fns)
    else:
        out[task][0] = env_cls(fns)
    return {name: dict(task_map) for name, task_map in out.items()}
