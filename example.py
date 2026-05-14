"""
Fish Audio S2-Pro (local) — Emotion & Prosody Demo

S2-Pro supports inline, free-form bracketed emotion and delivery tags, such as
`[whisper in small voice]`, `[laughing]`, and `[professional broadcast tone]`.

Prerequisites:
  1. uv run python download_models.py   # one-time model download
  2. uv run python server.py --bg       # start model server in background
  3. uv run python example.py           # run this demo

The demo saves output/emotion_demo.wav.
"""

import sys
from pathlib import Path

from tts_client import TTSParams, is_server_up, synthesize_to_file

OUTPUT_DIR = Path("output")

# ---------------------------------------------------------------------------
# Demo script — a compact sampler with varied S2-Pro inline tags
# ---------------------------------------------------------------------------
DEMO_SCRIPT = """\
[warm professional tone] This is a local Fish Audio S2-Pro quality check.
[whisper in small voice] Now I am speaking quietly, close to the microphone.
[angry, volume up] That was not the plan, and I need everyone to pay attention.
[sad, low voice] Some days are heavier than others, but we keep moving.
[laughing, delighted] Wait, that actually worked better than I expected.
[excited, pitch up] The local Streamlit path is ready for testing.
"""

# Emotion tags used — printed as a reference
TAGS_USED = [
    "[warm professional tone]",
    "[whisper in small voice]",
    "[angry, volume up]",
    "[sad, low voice]",
    "[laughing, delighted]",
    "[excited, pitch up]",
]


def main() -> None:
    if not is_server_up():
        print("Server is not running. Start it first:")
        print("  uv run python server.py --bg")
        sys.exit(1)

    print("=== Fish Audio S2-Pro — Emotion Demo ===\n")
    print(f"Script: {len(DEMO_SCRIPT)} characters")
    print(f"Emotion tags used ({len(TAGS_USED)}):")
    for t in TAGS_USED:
        print(f"  {t}")

    out_file = OUTPUT_DIR / "emotion_demo.wav"
    print(f"\nGenerating audio → {out_file} …")

    params = TTSParams(
        format="wav",
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=512,
    )
    synthesize_to_file(DEMO_SCRIPT, out_file, params, timeout=900)

    print(f"\nDone. Listen to {out_file}")


if __name__ == "__main__":
    main()
