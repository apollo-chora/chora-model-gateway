import logging
from typing import Any

import structlog

from .config import Settings, load_models
from .database import Database
from .grpc_server import start_grpc
from .http import create_app
from .service import Gateway

settings = Settings.load()
models = load_models(settings.registry_path)
db = Database(settings.database_url)
gateway = Gateway(settings, models, db)
app = create_app(gateway, db, models)
grpc_server: Any | None = None

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)


@app.on_start
async def start(_: Any) -> None:
    global grpc_server
    await db.start()
    grpc_server, _ = await start_grpc(gateway, settings.grpc_port)
    structlog.get_logger().info("gateway_boot", models=len(models), version=settings.service_version)


@app.on_stop
async def stop(_: Any) -> None:
    if grpc_server is not None:
        await grpc_server.stop(5)
    await db.close()
