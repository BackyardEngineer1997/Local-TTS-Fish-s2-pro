"""
Fish Audio S2-Pro — Streamlit UI (local inference)

Run:
  uv run streamlit run app.py

The app talks to the fish-speech API server running on localhost:8080.
If the server is not up, it shows how to start it.
"""

import io
import subprocess
import sys
import time

import streamlit as st

from tts_client import TTSParams, is_server_up, synthesize

# ---------------------------------------------------------------------------
# Emotion tag reference
# ---------------------------------------------------------------------------
EMOTION_REFERENCE = """
S2-Pro accepts free-form natural-language instructions in square brackets.

**Emotion**
`[super happy]` `[sad and tired]` `[angry]` `[nervous]` `[delighted]`

**Delivery**
`[whisper in small voice]` `[professional broadcast tone]` `[low voice]`
`[pitch up]` `[slow and reflective]` `[fast excited tone]`

**Vocal Effects**
`[laughing]` `[chuckle]` `[sigh]` `[inhale]` `[exhale]`
`[clearing throat]`

**Pause / Emphasis**
`[pause]` `[short pause]` `[emphasis]` `[volume up]` `[volume down]`

**Combos**
```
[sigh] I can't believe this.
[angry] How dare you!
[pause] Okay. Breathe.
[chuckle] Actually, kind of funny.
[whisper in small voice] Between us? I'm terrified.
```
"""

SAMPLE_SCRIPT = """\
[calm documentary narrator, measured pace] At 2:17 in the morning, the server finally answered.

[whispering, cautious, close to the microphone] Wait... did you hear that? Don't move.

[nervous laugh, trying to stay calm] Okay. That's fine. Totally fine. Probably just the wind.

[angry, sharp, volume up] No. Absolutely not. I told you not to touch that switch!

[sad, low voice, slow] I thought fixing it would feel like a victory. But now the room is quiet, and I miss the noise.

[delighted, laughing, bright voice] Oh! There it is! It worked! It actually worked!

[excited, fast pace, pitch up] Local S2-Pro is running, the Streamlit app is connected, and this voice finally has some range.
"""

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fish Audio S2-Pro — Local TTS",
    page_icon="Fish",
    layout="wide",
)

st.title("Fish Audio S2-Pro — Local TTS")

# ---------------------------------------------------------------------------
# Server status banner
# ---------------------------------------------------------------------------
server_ok = is_server_up()

if server_ok:
    st.success("Server running at http://127.0.0.1:8080", icon="✅")
else:
    st.error("Server is not running.", icon="🔴")
    with st.expander("How to start the server", expanded=True):
        st.code(
            "# 1. Download model weights (once)\n"
            "uv run python download_models.py\n\n"
            "# 2. Start the server in the background\n"
            "uv run python server.py --bg\n\n"
            "# 3. Refresh this page",
            language="bash",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start server now", disabled=False):
            with st.spinner("Starting server — this takes ~60 s for model loading …"):
                from server import start_background
                start_background()
                time.sleep(3)
            st.rerun()
    with col2:
        if st.button("🔄 Refresh status"):
            st.rerun()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Generation Settings")

    fmt = st.selectbox("Format", ["wav", "mp3"], index=0)

    st.divider()
    st.subheader("Sampling")
    temperature = st.slider(
        "Temperature (expressiveness)",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Higher = more varied and emotionally expressive delivery",
    )
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.05,
        help="Reduces word/phrase repetition. 1.2 is a good default.",
    )
    seed = st.number_input(
        "Seed (blank = random)",
        min_value=0,
        max_value=2**31,
        value=None,
        step=1,
        help="Fix a seed to get reproducible output",
    )

    st.divider()
    st.subheader("Emotion Tag Reference")
    st.markdown(EMOTION_REFERENCE)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([3, 2])

with col_input:
    st.subheader("Script")
    text = st.text_area(
        "Enter text with emotion tags",
        value=SAMPLE_SCRIPT,
        height=440,
        label_visibility="collapsed",
    )
    st.caption(f"{len(text)} characters")

    generate_btn = st.button(
        "🎙️ Generate Audio",
        type="primary",
        disabled=not server_ok or not text.strip(),
        use_container_width=True,
    )

with col_output:
    st.subheader("Output")

    if generate_btn:
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            params = TTSParams(
                format=fmt,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=int(seed) if seed is not None else None,
            )
            with st.spinner("Synthesising speech …"):
                try:
                    audio_bytes = synthesize(text, params)
                    st.session_state["audio"] = audio_bytes
                    st.session_state["fmt"] = fmt
                    st.success(f"Generated {len(audio_bytes) / 1024:.1f} KB")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if "audio" in st.session_state:
        audio_bytes = st.session_state["audio"]
        audio_fmt = st.session_state.get("fmt", "wav")
        mime = "audio/wav" if audio_fmt == "wav" else "audio/mpeg"

        st.audio(audio_bytes, format=mime)

        st.download_button(
            label=f"⬇️ Download .{audio_fmt}",
            data=io.BytesIO(audio_bytes),
            file_name=f"fish_tts.{audio_fmt}",
            mime=mime,
            use_container_width=True,
        )
        st.caption(f"{len(audio_bytes)/1024:.1f} KB · {audio_fmt.upper()}")

    else:
        if server_ok:
            st.info("Audio will appear here after generation.")
            st.markdown(
                "**Quick start:**\n"
                "1. Edit or paste your script (use emotion tags!)\n"
                "2. Hit **Generate Audio**\n"
                "3. Listen directly or download"
            )
        else:
            st.warning("Start the server first, then generate audio.")
