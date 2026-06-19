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

import os
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names

OBS_STATE_DIM = 8
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_CAMERAS = ["agentview", "robot0_eye_in_hand"]


def _camera_obs_key(camera_name: str) -> str:
    return f"{camera_name}_image"


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(float(den), 0.0, abs_tol=1e-10):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


class RobosuiteEnv(gym.Env):
    """LeRobot gym.Env wrapper for robosuite manipulation tasks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "Lift",
        robot: str = "Panda",
        camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        control_freq: int = 10,
        episode_length: int = 400,
        reward_shaping: bool = False,
        state_dim: int = OBS_STATE_DIM,
        action_dim: int = ACTION_DIM,
        episode_index: int = 0,
    ):
        super().__init__()
        self.task = task
        self.robot = robot
        self.camera_name = parse_camera_names(camera_name)
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.control_freq = control_freq
        self._max_episode_steps = episode_length
        self.reward_shaping = reward_shaping
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_index = int(episode_index)

        self._env: Any = None
        self._step_count = 0
        self.task_description = self.task

        images = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_name
        }

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type '{self.obs_type}'. Use 'pixels' or 'pixels_agent_pos'.")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def _register_task(self) -> None:
        """Hook for wrappers that register additional robosuite tasks."""

    def _ensure_env(self) -> None:
        if self._env is not None:
            return

        # robosuite 1.4 can fail while importing numba-cached helpers from an
        # editable/relocated environment. This preserves normal JIT behavior
        # when users set their own value, while making the wrapper robust here.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

        import robosuite as suite
        from robosuite.controllers import load_controller_config

        self._register_task()

        controller_config = load_controller_config(default_controller="OSC_POSE")
        self._env = suite.make(
            env_name=self.task,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera_name,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=self.control_freq,
            horizon=self._max_episode_steps,
            reward_shaping=self.reward_shaping,
        )

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> RobotObservation:
        images = {
            cam: raw_obs[_camera_obs_key(cam)].astype(np.uint8)
            for cam in self.camera_name
            if _camera_obs_key(cam) in raw_obs
        }

        if self.obs_type == "pixels":
            return {"pixels": images}

        eef_pos = raw_obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32))
        eef_quat = raw_obs.get("robot0_eef_quat", np.zeros(4, dtype=np.float32))
        eef_axisangle = _quat2axisangle(eef_quat)
        gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
        agent_pos = np.concatenate([eef_pos, eef_axisangle, gripper_qpos[:2]], axis=-1).astype(np.float32)
        if agent_pos.shape[0] < self.state_dim:
            agent_pos = np.pad(agent_pos, (0, self.state_dim - agent_pos.shape[0]))
        elif agent_pos.shape[0] > self.state_dim:
            agent_pos = agent_pos[: self.state_dim]

        return {"pixels": images, "agent_pos": agent_pos}

    def render(self) -> np.ndarray:
        self._ensure_env()
        assert self._env is not None
        return self._env.sim.render(
            camera_name=self.camera_name[0],
            height=self.observation_height,
            width=self.observation_width,
            depth=False,
        )[::-1]

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        self._step_count = 0
        raw_obs = self._env.reset()
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        raw_obs, reward, done, info = self._env.step(action.astype(np.float32))
        self._step_count += 1

        check_success = getattr(self._env, "_check_success", None)
        is_success = bool(info.get("success", False) or (check_success() if check_success else False))
        terminated = bool(done or is_success)
        truncated = self._step_count >= self._max_episode_steps
        info.update({"task": self.task, "done": bool(done), "is_success": is_success})

        observation = self._format_raw_obs(raw_obs)
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": is_success,
            }
            self.reset()

        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


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
) -> list[Callable[[], RobosuiteEnv]]:
    def _make_env(episode_index: int) -> RobosuiteEnv:
        return RobosuiteEnv(
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


def create_robosuite_envs(
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
        raise ValueError(f"Unexpected robosuite Lift kwargs: {sorted(gym_kwargs)}")

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
