"""
Patches the hardcoded system prompt in the installed fish_speech package
so the S2-Pro model actually follows square-bracket emotion/style tags.

Run once after installing dependencies:
  uv run python patch_fish_speech.py
"""

import sys
from pathlib import Path

OLD_PLAIN = 'text="convert the provided text to speech"'
NEW_PLAIN = (
    'text="You are an expressive voice actor. Convert the text to speech '
    "following all style, emotion, and delivery instructions enclosed in "
    "square brackets exactly — such as [whisper], [angry], [laughing], "
    '[sad], [excited], [sigh], [gasp], etc."'
)

OLD_REF = 'text="convert the provided text to speech reference to the following:\\n\\nText:\\n"'
NEW_REF = (
    'text="You are an expressive voice actor. Convert the text to speech '
    "following all style, emotion, and delivery instructions enclosed in "
    "square brackets exactly — such as [whisper], [angry], [laughing], "
    '[sad], [excited], [sigh], [gasp], etc. Reference speech:\\n\\nText:\\n"'
)


def find_inference_file() -> Path:
    for path in sys.path:
        candidate = (
            Path(path) / "fish_speech" / "models" / "text2semantic" / "inference.py"
        )
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find fish_speech/models/text2semantic/inference.py in sys.path. "
        "Is fish-speech installed in this environment?"
    )


def patch(file: Path) -> None:
    text = file.read_text()

    if NEW_PLAIN in text and NEW_REF in text:
        print(f"Already patched: {file}")
        return

    changed = False

    if OLD_PLAIN in text:
        text = text.replace(OLD_PLAIN, NEW_PLAIN)
        changed = True
        print("  Patched: plain (no-reference) system prompt")
    elif NEW_PLAIN not in text:
        print("  WARNING: plain system prompt not found — may already be patched or changed upstream")

    if OLD_REF in text:
        text = text.replace(OLD_REF, NEW_REF)
        changed = True
        print("  Patched: reference system prompt")
    elif NEW_REF not in text:
        print("  WARNING: reference system prompt not found — may already be patched or changed upstream")

    if changed:
        file.write_text(text)
        print(f"  Written: {file}")
    else:
        print(f"  No changes needed: {file}")


def main() -> None:
    print("Patching fish_speech system prompt for emotion tag support...")
    try:
        f = find_inference_file()
        print(f"Found: {f}")
        patch(f)
        print("\nDone. Restart the server for the change to take effect.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
