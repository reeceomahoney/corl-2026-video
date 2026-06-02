"""Arrange random episode clips from a LeRobot v3 dataset into a grid video.

Picks COLS*ROWS random episodes from ``reece-omahoney/pi05-libero-plus`` and tiles their
*scene* camera (``observation.images.image`` — the third-person agentview, not the
``image2`` wrist camera) into a single 3x3 grid clip.

Episodes are stored as time-slices of larger shared MP4s, so each clip is cut out
with ffmpeg using the per-episode [from, to] timestamps in the episode metadata
(same approach as save_episodes.py). Clips have different lengths, so each tile
holds its last frame (tpad) until the longest clip ends, keeping the grid full.

Usage:
    uv run python generate_episode_grid.py
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

# --- Configuration ----------------------------------------------------------

REPO = "reece-omahoney/pi05-libero-plus"

# The scene (third-person agentview) camera. The other video feature,
# observation.images.image2, is the gripper-mounted wrist camera.
SCENE_CAMERA = "observation.images.image"

# Grid shape. Source tiles are square (256x256); each is center-cropped so the
# full COLSxROWS mosaic is exactly 1280x720 (cells of 256x240 for 5x3).
COLS = 5
ROWS = 3
N_CLIPS = COLS * ROWS
SEED = 0  # fixed seed so the random pick is reproducible

OUTPUT_PATH = Path(__file__).parent / "outputs" / "episode_grid.mp4"

CRF = 18  # re-encode quality for the cut clips / final grid (lower = better)


# --- Dataset metadata -------------------------------------------------------

def load_info() -> dict:
    path = hf_hub_download(REPO, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def read_episode_meta(fs: HfFileSystem) -> list[dict]:
    """Read all meta/episodes parquet shards as a list of row dicts."""
    shards = fs.glob(f"datasets/{REPO}/meta/episodes/**/*.parquet")
    return pq.read_table(shards, filesystem=fs).to_pylist()


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


def extract_scene_clip(ep_row: dict, video_template: str, dst: Path) -> float:
    """Cut this episode's scene-camera clip; return its duration in seconds."""
    cam = SCENE_CAMERA
    chunk = ep_row[f"videos/{cam}/chunk_index"]
    file_idx = ep_row[f"videos/{cam}/file_index"]
    start = ep_row[f"videos/{cam}/from_timestamp"]
    end = ep_row[f"videos/{cam}/to_timestamp"]

    rel = video_template.format(video_key=cam, chunk_index=chunk, file_index=file_idx)
    src = Path(hf_hub_download(REPO, rel, repo_type="dataset"))
    cut_clip(src, start, end, dst)
    return end - start


# --- Grid assembly ----------------------------------------------------------

def build_grid(clips: list[Path], durations: list[float], dst: Path) -> None:
    """Tile the clips into a COLSxROWS mosaic, looping each until the grid ends.

    Every clip loops (``-stream_loop -1``) so each tile keeps playing rather than
    freezing; the whole grid is then trimmed to the longest clip's duration.
    """
    max_dur = max(durations)

    # Loop every input infinitely; the output -t caps the grid at max_dur.
    inputs: list[str] = []
    for clip in clips:
        inputs += ["-stream_loop", "-1", "-i", str(clip)]

    # Each cell is sized so the whole COLSxROWS mosaic is exactly 1280x720.
    # The source tiles are square (256x256), so center-crop each to the cell
    # size (here 256x240) before stacking — only vertical pixels are trimmed.
    cell_w = 1280 // COLS
    cell_h = 720 // ROWS

    # Normalise SAR/fps and center-crop to the cell size for xstack.
    pad_filters = [
        f"[{i}:v]fps=20,setsar=1,crop={cell_w}:{cell_h}[v{i}]"
        for i in range(len(clips))
    ]

    # xstack layout for an equal-cell COLSxROWS mosaic.
    positions = [
        f"{(idx % COLS) * cell_w}_{(idx // COLS) * cell_h}"
        for idx in range(N_CLIPS)
    ]
    layout = "|".join(positions)
    stack_inputs = "".join(f"[v{i}]" for i in range(N_CLIPS))
    filtergraph = (
        ";".join(pad_filters)
        + f";{stack_inputs}xstack=inputs={N_CLIPS}:layout={layout}[grid]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            *inputs,
            "-filter_complex", filtergraph,
            "-map", "[grid]",
            "-t", f"{max_dur:.3f}",
            "-c:v", "libx264", "-crf", str(CRF), "-preset", "fast",
            "-pix_fmt", "yuv420p", str(dst),
        ],
        check=True,
    )


# --- Main -------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    info = load_info()
    video_template = info["video_path"]
    total = info["total_episodes"]

    # rng.sample returns the picks in random order, so slot placement is random too.
    rng = random.Random(SEED)
    chosen = rng.sample(range(total), N_CLIPS)
    print(f"Chosen episodes in slot order (seed={SEED}): {chosen}")

    ep_meta = read_episode_meta(fs)
    by_index = {row["episode_index"]: row for row in ep_meta}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        durations: list[float] = []
        for slot, ep in enumerate(chosen):
            dst = tmp_dir / f"clip_{slot}.mp4"
            dur = extract_scene_clip(by_index[ep], video_template, dst)
            clips.append(dst)
            durations.append(dur)
            print(f"  slot {slot}: episode {ep} ({dur:.2f}s)")

        print(f"Building {COLS}x{ROWS} grid...")
        build_grid(clips, durations, OUTPUT_PATH)

    print(f"Wrote {OUTPUT_PATH.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
