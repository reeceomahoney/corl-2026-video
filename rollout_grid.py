"""Arrange random successful rollout clips from two datasets into a 2x2 grid video.

Picks 2 random *successful* episodes from each of ``reece-omahoney/rollout_remove-ethernet``
and ``reece-omahoney/rollout_remove-pen-lid`` and tiles their *scene* camera
(``observation.images.top`` — the third-person overhead view, not the wrist cameras)
into a single 2x2 grid clip.

An episode counts as a success if any of its frames has ``success == True`` (the sparse
terminal reward of a rollout), the same definition used by save_episodes.py.

Episodes are stored as time-slices of larger shared MP4s, so each clip is cut out with
ffmpeg using the per-episode [from, to] timestamps in the episode metadata. Clips have
different lengths, so each tile loops until the longest clip ends, keeping the grid full.

Usage:
    uv run python rollout_grid.py
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

# 2 random success clips are taken from each repo, in listed order.
REPOS = [
    "reece-omahoney/rollout_remove-ethernet",
    "reece-omahoney/rollout_remove-pen-lid",
]
CLIPS_PER_REPO = 2

# The scene (third-person overhead) camera. The other video features,
# observation.images.{left,right}_wrist, are the gripper-mounted wrist cameras.
SCENE_CAMERA = "observation.images.top"

# Grid shape. Cells are sized so the full COLSxROWS mosaic is exactly 1920x1080;
# each source tile is scaled to cover its cell and center-cropped (960x540 for 2x2).
COLS = 2
ROWS = 2
N_CLIPS = COLS * ROWS
CELL_W = 1920 // COLS
CELL_H = 1080 // ROWS

SEED = 0  # fixed seed so the random pick is reproducible

OUTPUT_PATH = Path(__file__).parent / "outputs" / "rollout_grid.mp4"

CRF = 18  # re-encode quality for the cut clips / final grid (lower = better)

EXTRA_SECONDS = 5.0  # extend the grid past the longest clip by looping this much


# --- Dataset metadata -------------------------------------------------------

def load_info(repo: str) -> dict:
    path = hf_hub_download(repo, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def read_episode_meta(fs: HfFileSystem, repo: str) -> list[dict]:
    """Read all meta/episodes parquet shards as a list of row dicts."""
    shards = fs.glob(f"datasets/{repo}/meta/episodes/**/*.parquet")
    return pq.read_table(shards, filesystem=fs).to_pylist()


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


def extract_scene_clip(repo: str, ep_row: dict, video_template: str, dst: Path) -> float:
    """Cut this episode's scene-camera clip; return its duration in seconds."""
    cam = SCENE_CAMERA
    chunk = ep_row[f"videos/{cam}/chunk_index"]
    file_idx = ep_row[f"videos/{cam}/file_index"]
    start = ep_row[f"videos/{cam}/from_timestamp"]
    end = ep_row[f"videos/{cam}/to_timestamp"]

    rel = video_template.format(video_key=cam, chunk_index=chunk, file_index=file_idx)
    src = Path(hf_hub_download(repo, rel, repo_type="dataset"))
    cut_clip(src, start, end, dst)
    return end - start


# --- Grid assembly ----------------------------------------------------------

def build_grid(clips: list[Path], durations: list[float], fps: float, dst: Path) -> None:
    """Tile the clips into a COLSxROWS mosaic, looping each until the grid ends.

    Every clip loops (``-stream_loop -1``) so each tile keeps playing rather than
    freezing; the whole grid is then trimmed to the longest clip's duration plus
    EXTRA_SECONDS of extra looping.
    """
    max_dur = max(durations) + EXTRA_SECONDS

    # Loop every input infinitely; the output -t caps the grid at max_dur.
    inputs: list[str] = []
    for clip in clips:
        inputs += ["-stream_loop", "-1", "-i", str(clip)]

    # Normalise fps/SAR, then scale each tile to cover its cell and center-crop
    # to the exact cell size so the assembled grid is precisely 1920x1080.
    pad_filters = [
        f"[{i}:v]fps={fps},"
        f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=increase,"
        f"crop={CELL_W}:{CELL_H},setsar=1[v{i}]"
        for i in range(len(clips))
    ]

    # xstack layout for an equal-cell COLSxROWS mosaic.
    positions = [
        f"{(idx % COLS) * CELL_W}_{(idx // COLS) * CELL_H}" for idx in range(N_CLIPS)
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
    rng = random.Random(SEED)

    # Pick CLIPS_PER_REPO random successful episodes from each repo.
    chosen_by_repo: list[tuple[str, list[int]]] = []
    fps = 20.0
    for repo in REPOS:
        info = load_info(repo)
        fps = info["fps"]
        success_by_ep = episode_success(fs, repo)
        successes = sorted(ep for ep, ok in success_by_ep.items() if ok)
        if len(successes) < CLIPS_PER_REPO:
            raise RuntimeError(
                f"{repo}: only {len(successes)} successful episodes "
                f"(need {CLIPS_PER_REPO})"
            )
        chosen = rng.sample(successes, CLIPS_PER_REPO)
        print(f"{repo}: {len(successes)} successes, chose {chosen}")
        chosen_by_repo.append((repo, chosen))

    diagonals = [[0, 3], [1, 2]]
    rng.shuffle(diagonals)  # randomise which repo takes which diagonal
    picks: list[tuple[str, int]] = [("", -1)] * N_CLIPS
    for (repo, chosen), slots in zip(chosen_by_repo, diagonals):
        rng.shuffle(slots)  # randomise corner within the diagonal
        for ep, slot in zip(chosen, slots):
            picks[slot] = (repo, ep)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        durations: list[float] = []
        ep_meta_cache: dict[str, dict[int, dict]] = {}
        for slot, (repo, ep) in enumerate(picks):
            if repo not in ep_meta_cache:
                meta = read_episode_meta(fs, repo)
                ep_meta_cache[repo] = {row["episode_index"]: row for row in meta}
            video_template = load_info(repo)["video_path"]
            dst = tmp_dir / f"clip_{slot}.mp4"
            dur = extract_scene_clip(
                repo, ep_meta_cache[repo][ep], video_template, dst
            )
            clips.append(dst)
            durations.append(dur)
            print(f"  slot {slot}: {repo} episode {ep} ({dur:.2f}s)")

        print(f"Building {COLS}x{ROWS} grid...")
        build_grid(clips, durations, fps, OUTPUT_PATH)

    print(f"Wrote {OUTPUT_PATH.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
