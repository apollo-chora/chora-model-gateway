# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project
COPY . .
RUN uv sync --no-dev
RUN uv run --with grpcio-tools python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/model_gateway_service.proto
FROM python:3.13-slim-bookworm
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CHORA_MODEL_REGISTRY=/app/config/models.yaml
EXPOSE 8080 9090
CMD ["granian","--interface","asgi","--host","0.0.0.0","--port","8080","app.gateway:app"]
