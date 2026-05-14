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
# Demo scripts — baseline/control pairs using official-style S2-Pro tags
# ---------------------------------------------------------------------------
CLIPS = [
    (
        "01_baseline_plain",
        "I cannot believe you actually solved the problem on the first try.",
    ),
    (
        "02_laugh",
        "I cannot believe you actually solved the problem on the first try. [laugh]",
    ),
    (
        "03_laughing_inline",
        "I tried to stay serious, [laughing] but that was the funniest bug report I have ever seen.",
    ),
    (
        "04_chuckle",
        "Yeah... [chuckle] okay, I admit it. That was a pretty clever workaround.",
    ),
    (
        "05_chuckling",
        "[chuckling] No, no, keep going. I want to see where this plan ends up.",
    ),
    (
        "06_clearing_throat",
        "[clearing throat] Ahem. Let us try that announcement one more time, from the top.",
    ),
    (
        "07_sigh",
        "[sigh] I know. We fixed one problem, and somehow found three more waiting behind it.",
    ),
    (
        "08_gasp",
        "Wait... [gasp] the lights just came back on.",
    ),
    (
        "09_inhale_exhale",
        "[inhale] Okay, let me think. [short pause] Yes. I know exactly what happened. [exhale]",
    ),
    (
        "10_whisper",
        "[whisper] Stay very quiet. I think the microphone is finally picking up the room.",
    ),
    (
        "11_angry",
        "[angry] Stop. I told you not to touch that switch. Put it back, right now.",
    ),
    (
        "12_excited",
        "[excited] It worked! The local server is running, Streamlit is connected, and the demo is ready.",
    ),
    (
        "13_sad",
        "[sad] I thought fixing it would feel like a victory, but now the room is quiet.",
    ),
    (
        "14_effect_chain",
        "[clearing throat] We are live. [inhale] Ready? [gasp] Wait, that actually worked. [laughing]",
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
