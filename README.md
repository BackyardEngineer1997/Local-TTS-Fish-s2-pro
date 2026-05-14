# Local Fish Audio S2-Pro TTS

This repo runs the open-source Fish Audio S2-Pro TTS model locally through a
small API server and a Streamlit frontend. It uses inline S2-Pro emotion and
delivery tags such as `[whisper in small voice]`, `[angry, volume up]`, and
`[laughing, delighted]`.

## What Changed

- Uses open-source `fishaudio/s2-pro` weights instead of the cloud API.
- Runs the local Fish Speech API server at `http://127.0.0.1:8080`.
- Downloads model files into `checkpoints/s2-pro/`.
- Streamlit talks to the local server through `tts_client.py`.
- `example.py` creates a compact WAV sample with multiple emotion tags.

## Local Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

Download the S2-Pro model weights:

```bash
uv run python download_models.py
```

Start the local server:

```bash
uv run python server.py --bg
```

Check server status:

```bash
uv run python server.py --status
```

Generate a test WAV:

```bash
uv run python example.py
```

The test file is written to:

```text
output/emotion_demo.wav
```

Start the Streamlit frontend:

```bash
uv run streamlit run app.py
```

Stop the background server:

```bash
uv run python server.py --stop
```

## Lightning.ai / GPU Setup

Yes, running this on Lightning.ai with a CUDA GPU should be much faster than
Apple Silicon MPS. The local MPS run successfully loaded S2-Pro, but generation
was slow enough that long samples can take many minutes. A CUDA GPU should cut
startup and token generation time substantially, especially on an L4, A10G,
A100, or similar.

On a Lightning.ai Studio or machine:

```bash
git clone <your-repo-url>
cd Local-TTS-Fish-s2-pro
sudo apt-get update
sudo apt-get install -y portaudio19-dev ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python download_models.py
uv run python server.py --bg
uv run python example.py
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The repo is set up so you can pull it on Lightning.ai and run those commands.
Do not commit the downloaded model weights; `checkpoints/` is ignored by git.
Download them directly on the GPU machine with `download_models.py`.

If dependency installation fails with `fatal error: portaudio.h: No such file or
directory`, run:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

Then rerun:

```bash
uv sync
uv run python download_models.py
```

## Emotion Tag Format

S2-Pro uses square-bracket natural language tags:

```text
[warm professional tone] This is a local quality check.
[whisper in small voice] Now I am speaking quietly.
[angry, volume up] That was not the plan.
[sad, low voice] Some days are heavier than others.
[laughing, delighted] Wait, that actually worked.
[excited, pitch up] The local Streamlit path is ready.
```

## Notes

- The first server startup loads several GB of S2-Pro weights and can take a few
  minutes.
- On MPS, generation is very slow. Use short prompts locally, or use CUDA for
  real testing.
- The server manager uses the local `local_api_server.py` wrapper to skip an
  expensive upstream warmup step that is painful on MPS.
- `server.log` contains model load and generation logs.
