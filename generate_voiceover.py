"""Generate a voiceover clip for each VO block in script.md using Piper (offline TTS).

For every `## §N — Title` section in the script, the `**VO:**` text is extracted,
synthesised to speech, and written to outputs/ as a separate WAV file. Before any
audio is generated, the per-section word count and estimated narration time (and the
total) are printed.

Usage:
    uv run python generate_voiceover.py
"""

from __future__ import annotations

import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice

# --- Configuration ----------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent / "script.md"
OUTPUT_DIR = Path(__file__).parent / "outputs"
VOICE_DIR = Path(__file__).parent / "voices"
VOICE_NAME = "en_US-lessac-high"
WORDS_PER_MINUTE = 150  # narration pace, per script.md


@dataclass
class Section:
    number: int
    title: str
    vo: str

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return f"{self.number:02d}_{slug}"

    @property
    def word_count(self) -> int:
        return len(self.vo.split())

    @property
    def est_seconds(self) -> float:
        return self.word_count / WORDS_PER_MINUTE * 60


# --- Parsing ----------------------------------------------------------------

SECTION_RE = re.compile(r"^## §(\d+)\s+—\s+(.+?)\s*$")


def parse_sections(markdown: str) -> list[Section]:
    """Extract (section number, title, VO text) for every section with a VO block."""
    sections: list[Section] = []
    number: int | None = None
    title = ""
    vo_lines: list[str] = []
    collecting = False

    def flush() -> None:
        if number is not None and vo_lines:
            text = " ".join(line.strip() for line in vo_lines).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                sections.append(Section(number, title, text))

    for line in markdown.splitlines():
        heading = SECTION_RE.match(line)
        if heading:
            flush()
            number = int(heading.group(1))
            title = heading.group(2)
            vo_lines = []
            collecting = False
            continue

        if line.lstrip().startswith("**VO:**"):
            collecting = True
            vo_lines = [line.split("**VO:**", 1)[1]]
            continue

        if collecting:
            # The VO block ends at the Shots block or a horizontal rule.
            if line.lstrip().startswith("**Shots:**") or line.strip().startswith("___"):
                collecting = False
                continue
            vo_lines.append(line)

    flush()
    return sections


# --- Reporting --------------------------------------------------------------

def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def print_estimates(sections: list[Section]) -> None:
    print(f"Pace: {WORDS_PER_MINUTE} wpm\n")
    print(f"{'Section':<28}{'Words':>8}{'Est. time':>12}")
    print("-" * 48)
    total_words = 0
    total_seconds = 0.0
    for s in sections:
        print(f"{f'§{s.number} {s.title}':<28}{s.word_count:>8}{format_time(s.est_seconds):>12}")
        total_words += s.word_count
        total_seconds += s.est_seconds
    print("-" * 48)
    print(f"{'TOTAL':<28}{total_words:>8}{format_time(total_seconds):>12}\n")


# --- Synthesis --------------------------------------------------------------

def load_voice() -> PiperVoice:
    VOICE_DIR.mkdir(exist_ok=True)
    model_path = VOICE_DIR / f"{VOICE_NAME}.onnx"
    if not model_path.exists():
        print(f"Downloading voice '{VOICE_NAME}' (one-time)...")
        download_voice(VOICE_NAME, VOICE_DIR)
    return PiperVoice.load(model_path)


def synthesize(voice: PiperVoice, section: Section) -> Path:
    out_path = OUTPUT_DIR / f"{section.slug}.wav"
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(section.vo, wav_file)
    return out_path


# --- Main -------------------------------------------------------------------

def main() -> int:
    if not SCRIPT_PATH.exists():
        print(f"Script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1

    sections = parse_sections(SCRIPT_PATH.read_text())
    if not sections:
        print("No VO blocks found in script.", file=sys.stderr)
        return 1

    print_estimates(sections)

    OUTPUT_DIR.mkdir(exist_ok=True)
    voice = load_voice()

    print("Generating voiceover clips...")
    for section in sections:
        out_path = synthesize(voice, section)
        print(f"  ✓ §{section.number} {section.title} -> {out_path.relative_to(Path.cwd())}")

    print(f"\nDone. {len(sections)} clips written to {OUTPUT_DIR.relative_to(Path.cwd())}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
