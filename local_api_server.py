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
import sys
from argparse import Namespace
from pathlib import Path
from threading import Lock

import pyrootutils
import uvicorn
from kui.asgi import Depends, FactoryClass, HTTPException, Kui, OpenAPI, Routes
from kui.cors import CORSConfig
from kui.openapi.specification import Info
from kui.security import bearer_auth
from loguru import logger
from typing_extensions import Annotated


def _local_setup_root(*args, pythonpath=False, cwd=False, **kwargs):
    root = Path(__file__).resolve().parent
    if pythonpath and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if cwd:
        os.chdir(root)
    return root


pyrootutils.setup_root = _local_setup_root

from tools.server.api_utils import MsgPackRequest, parse_args
from tools.server.exception_handler import ExceptionHandler
from tools.server.model_manager import ModelManager
from tools.server.views import routes

ENV_ARGS_KEY = "FISH_API_SERVER_ARGS"


def _skip_warm_up(self: ModelManager, tts_inference_engine) -> None:
    logger.info("Skipping upstream 1024-token warmup for local MPS startup.")


ModelManager.warm_up = _skip_warm_up


class API(ExceptionHandler):
    def __init__(self, args: Namespace | None = None):
        self.args = args or parse_args()

        def api_auth(endpoint):
            async def verify(token: Annotated[str, Depends(bearer_auth)]):
                if token != self.args.api_key:
                    raise HTTPException(401, None, "Invalid token")
                return await endpoint()

            async def passthrough():
                return await endpoint()

            if self.args.api_key is not None:
                return verify
            return passthrough

        self.routes = Routes(routes, http_middlewares=[api_auth])
        self.openapi = OpenAPI(
            Info(
                {
                    "title": "Fish Speech API",
                    "version": "2.0.0",
                }
            ),
        ).routes

        self.app = Kui(
            routes=self.routes + self.openapi[1:],
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config=CORSConfig(),
        )

        self.app.state.lock = Lock()
        self.app.state.device = self.args.device
        self.app.state.max_text_length = self.args.max_text_length
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        app.state.model_manager = ModelManager(
            mode=self.args.mode,
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Startup done, listening server at http://{self.args.listen}")


def create_app():
    args_env = os.environ.get(ENV_ARGS_KEY)
    args = None

    if args_env:
        try:
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
