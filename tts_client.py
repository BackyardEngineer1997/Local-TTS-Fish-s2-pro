"""
HTTP client for the local fish-speech API server.

Wraps the msgpack protocol so callers just deal with text → bytes.
"""

from dataclasses import dataclass, field
from pathlib import Path

import ormsgpack
import requests
from fish_speech.utils.schema import ServeTTSRequest

SERVER_URL = "http://127.0.0.1:8080"


@dataclass
class TTSParams:
    format: str = "wav"
    temperature: float = 0.85
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    chunk_length: int = 200
    max_new_tokens: int = 1024
    seed: int | None = None


def synthesize(text: str, params: TTSParams | None = None, timeout: int = 300) -> bytes:
    """Call the local server and return raw audio bytes."""
    if params is None:
        params = TTSParams()

    req = ServeTTSRequest(
        text=text,
        format=params.format,
        temperature=params.temperature,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
        chunk_length=params.chunk_length,
        max_new_tokens=params.max_new_tokens,
        seed=params.seed,
        normalize=True,
    )

    response = requests.post(
        f"{SERVER_URL}/v1/tts",
        data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={"content-type": "application/msgpack"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


def is_server_up(timeout: float = 1.0) -> bool:
    try:
        return requests.get(f"{SERVER_URL}/v1/health", timeout=timeout).ok
    except Exception:
        return False


def synthesize_to_file(
    text: str,
    output_path: Path,
    params: TTSParams | None = None,
    timeout: int = 300,
) -> None:
    audio = synthesize(text, params, timeout=timeout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio)
    size_kb = len(audio) / 1024
    print(f"  Saved {size_kb:.1f} KB → {output_path}")
