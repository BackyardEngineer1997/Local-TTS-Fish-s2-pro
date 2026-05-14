"""
Local launcher for Fish Speech's API server.

Upstream S2-Pro warms up with max_new_tokens=1024, which can take many minutes
on Apple Silicon MPS. This wrapper keeps the upstream routes and model manager
but skips that warmup so local startup reaches /v1/health promptly.
"""

import json
import multiprocessing
import os
import re

import uvicorn
from loguru import logger

from tools.api_server import API, ENV_ARGS_KEY
from tools.server.api_utils import parse_args
from tools.server.model_manager import ModelManager


def _skip_warm_up(self: ModelManager, tts_inference_engine) -> None:
    logger.info("Skipping upstream 1024-token warmup for local MPS startup.")


ModelManager.warm_up = _skip_warm_up


def create_app():
    args_env = os.environ.get(ENV_ARGS_KEY)
    args = None

    if args_env:
        try:
            from argparse import Namespace

            args = Namespace(**json.loads(args_env))
        except Exception as exc:
            logger.warning(f"Failed to load args from {ENV_ARGS_KEY}: {exc}")

    return API(args=args).app


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    os.environ[ENV_ARGS_KEY] = json.dumps(vars(args))

    match = re.search(r"\[([^\]]+)\]:(\d+)$", args.listen)
    if match:
        host, port = match.groups()
    else:
        host, port = args.listen.split(":")

    uvicorn.run(
        "local_api_server:create_app",
        host=host,
        port=int(port),
        workers=args.workers,
        log_level="info",
        factory=True,
    )
