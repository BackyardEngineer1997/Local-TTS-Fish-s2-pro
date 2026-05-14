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
# Demo script — a compact scene with strong contrast and varied S2-Pro tags
# ---------------------------------------------------------------------------
DEMO_SCRIPT = """\
[calm documentary narrator, measured pace] At 2:17 in the morning, the server finally answered.

[whispering, cautious, close to the microphone] Wait... did you hear that? Don't move.

[nervous laugh, trying to stay calm] Okay. That's fine. Totally fine. Probably just the wind.

[angry, sharp, volume up] No. Absolutely not. I told you not to touch that switch!

[sad, low voice, slow] I thought fixing it would feel like a victory. But now the room is quiet, and I miss the noise.

[delighted, laughing, bright voice] Oh! There it is! It worked! It actually worked!

[excited, fast pace, pitch up] Local S2-Pro is running, the Streamlit app is connected, and this voice finally has some range.
"""

# Emotion tags used — printed as a reference
TAGS_USED = [
    "[calm documentary narrator, measured pace]",
    "[whispering, cautious, close to the microphone]",
    "[nervous laugh, trying to stay calm]",
    "[angry, sharp, volume up]",
    "[sad, low voice, slow]",
    "[delighted, laughing, bright voice]",
    "[excited, fast pace, pitch up]",
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
        max_new_tokens=768,
    )
    synthesize_to_file(DEMO_SCRIPT, out_file, params, timeout=900)

    print(f"\nDone. Listen to {out_file}")


if __name__ == "__main__":
    main()
