"""
Fish Audio S2-Pro (local) — Emotion & Prosody Demo

S2-Pro supports inline, free-form bracketed emotion and delivery tags, such as
`[whisper in small voice]`, `[laughing]`, and `[professional broadcast tone]`.

Prerequisites:
  1. uv run python download_models.py   # one-time model download
  2. uv run python server.py --bg       # start model server in background
  3. uv run python example.py           # run this demo

The demo saves individual style clips plus output/emotion_demo_combined.wav.
"""

import sys
from pathlib import Path

from tts_client import TTSParams, is_server_up, synthesize_to_file

OUTPUT_DIR = Path("output")

# ---------------------------------------------------------------------------
# Demo scripts — separate clips make tag effects easier to judge
# ---------------------------------------------------------------------------
CLIPS = [
    (
        "01_neutral_narrator",
        "[calm documentary narrator, measured pace] The backup generator started at 2:17 in the morning.",
    ),
    (
        "02_whisper",
        "[whispering, very quiet, close to the microphone] Wait... stay still. I think someone is outside the door.",
    ),
    (
        "03_angry",
        "[angry, sharp, volume up] Stop. I told you not to touch that switch. Put it back, right now.",
    ),
    (
        "04_sad",
        "[sad, low voice, slow pace] I thought fixing it would feel like a victory. But now the room is quiet.",
    ),
    (
        "05_delighted",
        "[delighted, laughing, bright voice] Oh! There it is! It worked! I cannot believe that actually worked.",
    ),
    (
        "06_excited",
        "[excited, fast pace, pitch up] Local S2-Pro is running, Streamlit is connected, and the demo is ready.",
    ),
    (
        "07_laughing",
        "[laughing, amused, bright voice] I tried to stay serious, but that was the funniest thing I heard all week.",
    ),
    (
        "08_chuckle",
        "[soft chuckle, warm voice] Yeah... okay, I admit it. That was a pretty clever workaround.",
    ),
    (
        "09_sigh",
        "[deep sigh, tired voice] I know. We fixed one problem, and somehow found three more waiting behind it.",
    ),
    (
        "10_gasp",
        "[gasp, surprised, breathy voice] Oh! Wait. The lights just came back on.",
    ),
    (
        "11_whisper_laugh",
        "[whispering, quiet laugh, close to the microphone] Don't laugh... but I think the dramatic button was just the power switch.",
    ),
    (
        "12_cough",
        "[coughing, clears throat, slightly embarrassed] Sorry. Give me one second. I think I swallowed that sentence sideways.",
    ),
    (
        "13_throat_clear",
        "[clearing throat, formal voice] Ahem. Let us try that announcement one more time, from the top.",
    ),
    (
        "14_breathless",
        "[breathless, hurried, slightly panicked] I ran all the way here because the server finally came online.",
    ),
    (
        "15_giggle",
        "[giggle, playful, trying not to laugh] No, no, I'm fine. It's just... the error message was weirdly dramatic.",
    ),
]

COMBINED_SCRIPT = "\n\n".join(text for _, text in CLIPS)


def main() -> None:
    if not is_server_up():
        print("Server is not running. Start it first:")
        print("  uv run python server.py --bg")
        sys.exit(1)

    print("=== Fish Audio S2-Pro — Emotion Demo ===\n")
    print(f"Generating {len(CLIPS)} individual style clips plus one combined file.")

    params = TTSParams(
        format="wav",
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=384,
    )

    for name, text in CLIPS:
        out_file = OUTPUT_DIR / f"{name}.wav"
        print(f"\nGenerating {name} -> {out_file}")
        synthesize_to_file(text, out_file, params, timeout=600)

    combined_file = OUTPUT_DIR / "emotion_demo_combined.wav"
    print(f"\nGenerating combined scene -> {combined_file}")
    combined_params = TTSParams(
        format="wav",
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=768,
    )
    synthesize_to_file(COMBINED_SCRIPT, combined_file, combined_params, timeout=900)

    print(f"\nDone. Listen to the WAV files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
