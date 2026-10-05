FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /app
# chora-contracts is a sibling path dependency (pyproject: path = "../chora-contracts").
# It is supplied as a named build context and staged at /chora-contracts so the
# relative path resolves inside the image. Only the files the wheel needs are
# copied: the pyproject, src/chora_contracts, and gen/python (force-included as
# chora_contracts_gen).
COPY --from=chora-contracts pyproject.toml /chora-contracts/pyproject.toml
COPY --from=chora-contracts src /chora-contracts/src
COPY --from=chora-contracts gen/python /chora-contracts/gen/python
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project
COPY . .
RUN uv sync --no-dev
RUN uv run --with grpcio-tools python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/model_gateway_service.proto
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CHORA_MODEL_REGISTRY=/app/config/models.yaml
EXPOSE 8080 9090
CMD ["granian","--interface","asgi","--host","0.0.0.0","--port","8080","app.gateway:app"]
