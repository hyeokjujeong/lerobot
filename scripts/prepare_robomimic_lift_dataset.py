#!/usr/bin/env python
"""Prepare yananchen/robomimic_lift for LeRobot diffusion training.

The source dataset uses short feature names such as `image`, `wrist_image`,
`state`, and `actions`. LeRobot policies expect canonical keys:

    image       -> observation.images.agentview
    wrist_image -> observation.images.robot0_eye_in_hand
    state       -> observation.state
    actions     -> action

This script downloads/copies the dataset and rewrites metadata plus parquet
column names in-place under the output root.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "yananchen/robomimic_lift"
DEFAULT_MAPPING = {
    "image": "observation.images.agentview",
    "wrist_image": "observation.images.robot0_eye_in_hand",
    "state": "observation.state",
    "actions": "action",
    "observation.images.image": "observation.images.agentview",
    "observation.images.wrist_image": "observation.images.robot0_eye_in_hand",
}


def parse_mapping(items: list[str] | None) -> dict[str, str]:
    mapping = dict(DEFAULT_MAPPING)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Mapping must be OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Mapping must be OLD=NEW, got: {item}")
        mapping[old] = new
    return mapping


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


def rewrite_keyed_dict(obj: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in obj.items():
        rewritten[mapping.get(key, key)] = value
    return rewritten


def rewrite_metadata(root: Path, mapping: dict[str, str], repo_id: str) -> None:
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = load_json(info_path)
        if "features" in info:
            info["features"] = rewrite_keyed_dict(info["features"], mapping)
        info["robot_type"] = info.get("robot_type") or "panda"
        dump_json(info_path, info)

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        dump_json(stats_path, rewrite_keyed_dict(stats, mapping))

    readme_path = root / "README.md"
    if readme_path.exists():
        text = readme_path.read_text()
        text += (
            "\n\n## Local training preparation\n\n"
            f"This local copy was prepared from `{repo_id}` with feature keys renamed for "
            "LeRobot robosuite inference.\n"
        )
        readme_path.write_text(text)


def rewrite_parquet_columns(root: Path, mapping: dict[str, str]) -> int:
    count = 0
    for parquet_path in sorted(root.rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        old_names = table.schema.names
        new_names = [mapping.get(name, name) for name in old_names]
        if new_names == old_names:
            continue
        table = table.rename_columns(new_names)
        pq.write_table(table, parquet_path)
        count += 1
    return count


def copy_or_download_source(repo_id: str, source_root: Path | None, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    if source_root is None:
        downloaded = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
        source_root = downloaded

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    ignore = shutil.ignore_patterns(".git", ".cache", "__pycache__")
    shutil.copytree(source_root, output_root, ignore=ignore)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--source-root", type=Path, default=None, help="Use an existing local dataset copy.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/robomimic_lift_lerobot"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--map",
        action="append",
        default=None,
        help="Extra/override feature mapping as OLD=NEW. Can be passed multiple times.",
    )
    args = parser.parse_args()

    mapping = parse_mapping(args.map)
    output_root = args.output_root.expanduser().resolve()

    copy_or_download_source(args.repo_id, args.source_root, output_root, args.overwrite)
    rewrite_metadata(output_root, mapping, args.repo_id)
    changed = rewrite_parquet_columns(output_root, mapping)

    print(f"Prepared dataset at: {output_root}")
    print(f"Rewrote parquet files: {changed}")
    print("Feature mapping:")
    for old, new in mapping.items():
        print(f"  {old} -> {new}")


if __name__ == "__main__":
    main()
