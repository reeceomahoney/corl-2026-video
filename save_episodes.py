"""Save one success and one failure episode (per camera) from rollout datasets.

For each LeRobot v3 rollout dataset listed in DATASETS, this picks the first
successful and the first failed episode, then writes one MP4 per camera for each
to ``outputs/`` in a nested layout::

    outputs/{dataset}/{success|failure}/{camera}.mp4

Episodes are stored as time-slices of larger shared MP4 files, so each clip is
cut out with ffmpeg using the per-episode [from, to] timestamps recorded in the
dataset's episode metadata. An episode counts as a success if any of its frames
has ``success == True`` (the sparse terminal reward of a rollout).

Usage:
    uv run python save_episodes.py
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

# --- Configuration ----------------------------------------------------------

DATASETS = {
    "remove-pen-lid": "reece-omahoney/rollout_remove-pen-lid",
    "remove-ethernet": "reece-omahoney/rollout_remove-ethernet",
}

BASE_DATASETS = {
    "remove-ethernet": ("reece-omahoney/remove-ethernet-2", 0),
    "remove-pen-lid": ("reece-omahoney/remove-pen-lid-2", 0),
}

OUTPUT_DIR = Path(__file__).parent / "outputs"

# Re-encode quality for the cut clips (lower CRF = higher quality / bigger file).
CRF = 18


# --- Dataset metadata -------------------------------------------------------

@dataclass
class Episode:
    index: int
    success: bool


def load_info(repo: str) -> dict:
    """Download and parse meta/info.json."""
    path = hf_hub_download(repo, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def camera_keys(info: dict) -> list[str]:
    """Return the feature keys that are stored as video (one per camera)."""
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def read_episode_meta(fs: HfFileSystem, repo: str) -> "pq.Table":
    """Read all meta/episodes parquet shards as a single table."""
    shards = fs.glob(f"datasets/{repo}/meta/episodes/**/*.parquet")
    return pq.read_table(shards, filesystem=fs)


def episode_success(fs: HfFileSystem, repo: str) -> dict[int, bool]:
    """Map episode_index -> success by reading only the needed data columns.

    Reading just two columns over the HF filesystem fetches the relevant parquet
    column chunks via HTTP range requests rather than the whole (~100 MB) shard.
    """
    shards = fs.glob(f"datasets/{repo}/data/**/*.parquet")
    table = pq.read_table(
        shards, columns=["episode_index", "success"], filesystem=fs
    )
    result: dict[int, bool] = {}
    for ep, ok in zip(
        table.column("episode_index").to_pylist(),
        table.column("success").to_pylist(),
    ):
        result[ep] = result.get(ep, False) or bool(ok)
    return result


def pick_episodes(success_by_ep: dict[int, bool]) -> dict[str, int]:
    """Pick the first successful and first failed episode by index."""
    chosen: dict[str, int] = {}
    for ep in sorted(success_by_ep):
        label = "success" if success_by_ep[ep] else "failure"
        if label not in chosen:
            chosen[label] = ep
    return chosen


# --- Video extraction -------------------------------------------------------

def cut_clip(src: Path, start: float, end: float, dst: Path) -> None:
    """Extract [start, end] of ``src`` into ``dst``, re-encoding for accuracy."""
    duration = end - start
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-crf", str(CRF), "-preset", "fast",
            "-pix_fmt", "yuv420p", "-an", str(dst),
        ],
        check=True,
    )


def save_episode_videos(
    repo: str,
    name: str,
    label: str,
    ep_row: dict,
    cameras: list[str],
    video_template: str,
) -> None:
    """Download each camera's shared MP4 and cut out this episode's clip."""
    ep_index = ep_row["episode_index"]
    for cam in cameras:
        chunk = ep_row[f"videos/{cam}/chunk_index"]
        file_idx = ep_row[f"videos/{cam}/file_index"]
        start = ep_row[f"videos/{cam}/from_timestamp"]
        end = ep_row[f"videos/{cam}/to_timestamp"]

        rel = video_template.format(
            video_key=cam, chunk_index=chunk, file_index=file_idx
        )
        src = Path(hf_hub_download(repo, rel, repo_type="dataset"))

        cam_short = cam.replace("observation.images.", "")
        dst_dir = OUTPUT_DIR / name / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{cam_short}.mp4"
        cut_clip(src, start, end, dst)
        print(f"  wrote {dst.relative_to(OUTPUT_DIR.parent)}")


# --- Main -------------------------------------------------------------------

def process_dataset(name: str, repo: str, fs: HfFileSystem) -> None:
    print(f"== {name} ({repo})")
    info = load_info(repo)
    cameras = camera_keys(info)
    video_template = info["video_path"]

    success_by_ep = episode_success(fs, repo)
    chosen = pick_episodes(success_by_ep)
    if "success" not in chosen:
        print("  WARNING: no successful episode found")
    if "failure" not in chosen:
        print("  WARNING: no failed episode found")

    ep_meta = read_episode_meta(fs, repo).to_pylist()
    by_index = {row["episode_index"]: row for row in ep_meta}

    for label, ep_index in chosen.items():
        print(f"  {label}: episode {ep_index}")
        save_episode_videos(
            repo, name, label, by_index[ep_index], cameras, video_template
        )


def process_base_dataset(name: str, repo: str, ep_index: int, fs: HfFileSystem) -> None:
    """Save a single, fixed episode from a base (training) dataset."""
    print(f"== {name} base ({repo})")
    info = load_info(repo)
    cameras = camera_keys(info)
    video_template = info["video_path"]

    ep_meta = read_episode_meta(fs, repo).to_pylist()
    by_index = {row["episode_index"]: row for row in ep_meta}
    if ep_index not in by_index:
        print(f"  WARNING: episode {ep_index} not found")
        return

    print(f"  base: episode {ep_index}")
    save_episode_videos(
        repo, name, "base", by_index[ep_index], cameras, video_template
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for name, repo in DATASETS.items():
        process_dataset(name, repo, fs)
    for name, (repo, ep_index) in BASE_DATASETS.items():
        process_base_dataset(name, repo, ep_index, fs)
    print("Done.")


if __name__ == "__main__":
    main()
