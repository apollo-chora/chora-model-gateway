import logging,structlog
from .config import Settings,load_models
from .database import Database
from .service import Gateway
from .http import create_app
from .grpc_server import start_grpc
settings=Settings.load();models=load_models(settings.registry_path);db=Database(settings.database_url);gateway=Gateway(settings,models,db);app=create_app(gateway,db,models)
structlog.configure(processors=[structlog.contextvars.merge_contextvars,structlog.processors.add_log_level,structlog.processors.TimeStamper(fmt="iso"),structlog.processors.JSONRenderer()],wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
@app.on_start
async def start(_):
 await db.start();app.services.grpc_server=await start_grpc(gateway,settings.grpc_port);structlog.get_logger().info("gateway_boot",models=len(models),version=settings.service_version)
@app.on_stop
async def stop(_):
 if getattr(app.services,"grpc_server",None):await app.services.grpc_server.stop(5)
 await db.close()
