#!/usr/bin/env python
"""Convert MimicGen robomimic HDF5 data to LeRobot format.

The defaults target Coffee_D2 for backwards compatibility, but the converter
can be reused for other MimicGen datasets with the same robomimic observation
keys by changing --input, --output-root, --repo-id, and --task.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

DEFAULT_INPUT = Path("/PublicSSD/jhri626/datasets/coffee_d2.hdf5")
DEFAULT_OUTPUT = Path("/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot")
DEFAULT_REPO_ID = "local/mimicgen_coffee_d2"
DEFAULT_TASK = "Coffee_D2"
DEFAULT_CAMERAS = {
    "agentview_image": "observation.images.agentview",
    "robot0_eye_in_hand_image": "observation.images.robot0_eye_in_hand",
}
STATE_MODES = ("axis_angle", "quat")


def quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if math.isclose(float(den), 0.0, abs_tol=1e-10):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def parse_camera_map(raw_maps: list[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for raw_map in raw_maps:
        if "=" not in raw_map:
            raise ValueError(
                f"Invalid camera map '{raw_map}'. Expected HDF5_KEY=LEROBOT_KEY, "
                "for example agentview_image=observation.images.agentview."
            )
        hdf5_key, lerobot_key = raw_map.split("=", 1)
        hdf5_key = hdf5_key.strip()
        lerobot_key = lerobot_key.strip()
        if not hdf5_key or not lerobot_key:
            raise ValueError(f"Invalid camera map '{raw_map}'. Keys cannot be empty.")
        cameras[hdf5_key] = lerobot_key
    return cameras


def make_agent_pos(obs: h5py.Group, frame_index: int, state_dim: int, state_mode: str) -> np.ndarray:
    eef_pos = obs["robot0_eef_pos"][frame_index].astype(np.float32)
    eef_quat = obs["robot0_eef_quat"][frame_index].astype(np.float32)
    if state_mode == "axis_angle":
        eef_rotation = quat_xyzw_to_axisangle(eef_quat)
    elif state_mode == "quat":
        eef_rotation = eef_quat
    else:
        raise ValueError(f"Unsupported state_mode '{state_mode}'. Expected one of {STATE_MODES}.")
    gripper_qpos = obs["robot0_gripper_qpos"][frame_index].astype(np.float32)
    agent_pos = np.concatenate([eef_pos, eef_rotation, gripper_qpos[:2]], axis=-1).astype(np.float32)
    if agent_pos.shape[0] < state_dim:
        agent_pos = np.pad(agent_pos, (0, state_dim - agent_pos.shape[0]))
    elif agent_pos.shape[0] > state_dim:
        agent_pos = agent_pos[:state_dim]
    return agent_pos


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def read_env_args(data_group: h5py.Group) -> dict[str, Any]:
    raw = data_group.attrs.get("env_args")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def infer_fps(data_group: h5py.Group, fallback: int) -> int:
    env_args = read_env_args(data_group)
    return int(env_args.get("env_kwargs", {}).get("control_freq", fallback))


def infer_image_shape(first_demo: h5py.Group, camera_key: str) -> tuple[int, int, int]:
    shape = first_demo["obs"][camera_key].shape
    return int(shape[1]), int(shape[2]), int(shape[3])


def build_features(
    first_demo: h5py.Group,
    action_dim: int,
    state_dim: int,
    use_videos: bool,
    camera_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": [f"state_{i}" for i in range(state_dim)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": [f"action_{i}" for i in range(action_dim)]},
        },
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for hdf5_key, lerobot_key in camera_map.items():
        features[lerobot_key] = {
            "dtype": image_dtype,
            "shape": infer_image_shape(first_demo, hdf5_key),
            "names": ["height", "width", "channel"],
        }
    return features


def validate_hdf5_keys(first_demo: h5py.Group, camera_map: dict[str, str]) -> None:
    required_obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", *camera_map.keys()]
    missing_obs_keys = [key for key in required_obs_keys if key not in first_demo["obs"]]
    missing_demo_keys = [key for key in ["actions"] if key not in first_demo]
    if missing_obs_keys or missing_demo_keys:
        available_obs = ", ".join(sorted(first_demo["obs"].keys()))
        available_demo = ", ".join(sorted(first_demo.keys()))
        raise KeyError(
            "HDF5 does not match the expected MimicGen robomimic layout. "
            f"Missing demo keys={missing_demo_keys}, missing obs keys={missing_obs_keys}. "
            f"Available demo keys=[{available_demo}], available obs keys=[{available_obs}]"
        )


def convert_dataset(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    raw_camera_map = args.camera_map or [
        f"{hdf5_key}={lerobot_key}" for hdf5_key, lerobot_key in DEFAULT_CAMERAS.items()
    ]
    camera_map = parse_camera_map(raw_camera_map)

    with h5py.File(input_path, "r") as h5:
        data = h5["data"]
        demo_keys = sorted_demo_keys(data)
        if args.max_episodes is not None:
            demo_keys = demo_keys[: args.max_episodes]
        if not demo_keys:
            raise ValueError("No demos found in HDF5 file.")

        first_demo = data[demo_keys[0]]
        validate_hdf5_keys(first_demo, camera_map)
        fps = args.fps or infer_fps(data, fallback=20)
        features = build_features(
            first_demo=first_demo,
            action_dim=int(first_demo["actions"].shape[-1]),
            state_dim=args.state_dim,
            use_videos=not args.no_videos,
            camera_map=camera_map,
        )

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_root,
            robot_type="panda",
            fps=fps,
            features=features,
            use_videos=not args.no_videos,
            video_backend=args.video_backend,
            vcodec=args.vcodec,
            batch_encoding_size=args.batch_encoding_size,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )

        start_time = time.time()
        for episode_index, demo_key in enumerate(demo_keys):
            demo = data[demo_key]
            obs = demo["obs"]
            episode_length = int(demo["actions"].shape[0])
            elapsed = time.time() - start_time
            d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)
            logging.info(
                "%s / %s episodes processed after %sd %sh %sm %.1fs",
                episode_index,
                len(demo_keys),
                d,
                h,
                m,
                s,
            )

            for frame_index in range(episode_length):
                reward = float(demo["rewards"][frame_index]) if "rewards" in demo else 0.0
                frame = {
                    "observation.state": make_agent_pos(obs, frame_index, args.state_dim, args.state_mode),
                    "action": demo["actions"][frame_index].astype(np.float32),
                    "reward": np.array([reward], dtype=np.float32),
                    "is_success": np.array([frame_index == episode_length - 1], dtype=bool),
                    "task": args.task,
                }
                for hdf5_key, lerobot_key in camera_map.items():
                    frame[lerobot_key] = obs[hdf5_key][frame_index]
                dataset.add_frame(frame)
            dataset.save_episode()

        dataset.finalize()

    logging.info("Prepared dataset at %s", output_root)
    logging.info(
        "repo_id=%s task=%s fps=%s episodes=%s state_mode=%s camera_map=%s",
        args.repo_id,
        args.task,
        fps,
        len(demo_keys),
        args.state_mode,
        camera_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=None, help="Defaults to HDF5 env_args control_freq.")
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Defaults to 8 for --state-mode axis_angle and 9 for --state-mode quat.",
    )
    parser.add_argument(
        "--state-mode",
        choices=STATE_MODES,
        default="axis_angle",
        help=(
            "How to store robot0_eef_quat inside observation.state. "
            "axis_angle stores pos(3)+rotvec(3)+gripper(2), matching the current MimicGen env wrapper. "
            "quat stores pos(3)+quat_xyzw(4)+gripper(2), usually with --state-dim=9."
        ),
    )
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        help=(
            "Camera mapping from HDF5 obs key to LeRobot key. Can be passed multiple times. "
            "Passing any --camera-map replaces the default Coffee_D2 camera mapping. "
            "Example: --camera-map agentview_image=observation.images.agentview"
        ),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store image columns instead of MP4 videos.")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.state_dim is None:
        args.state_dim = 8 if args.state_mode == "axis_angle" else 9

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    convert_dataset(args)


if __name__ == "__main__":
    main()
