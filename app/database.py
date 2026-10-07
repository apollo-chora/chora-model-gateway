from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
from google.protobuf.timestamp_pb2 import Timestamp

from .contracts import EventEnvelope, TokenUsageRecorded

SOURCE_SERVICE = "chora-model-gateway"
SCHEMA_VERSION = 1
# Bandit's B105 heuristic matches the word "TOKEN" in the constant NAME; these
# are NATS event type/topic identifiers, not credentials.
TOKEN_USAGE_EVENT_TYPE = "chora.observability.token_usage.recorded.v1"  # nosec B105
TOKEN_USAGE_TOPIC = "chora.observability.token_usage.recorded.v1"  # nosec B105


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp


def _build_outbox(event: dict[str, Any], occurred_at: datetime) -> tuple[bytes, dict[str, str]]:
    """Serialise the canonical TokenUsageRecorded payload and its flat
    snake_case outbox envelope.

    Mirrors the original Go gateway's buildTokenUsagePayload +
    buildOutboxEnvelope so observability's dispatcher and consumer read exactly
    what they always have. The gateway does not own this event shape —
    chora-contracts does (see app/contracts.py)."""
    usage_id = event["usage_id"]
    record = TokenUsageRecorded(
        envelope=EventEnvelope(
            event_id=usage_id,
            idempotency_key=usage_id,
            tenant_id=event["tenant_id"],
            gcid=event["gcid"],
            occurred_at=_timestamp(occurred_at),
            traceparent=event.get("traceparent") or "",
            tracestate=event.get("tracestate") or "",
            source_service=SOURCE_SERVICE,
            schema_version=SCHEMA_VERSION,
        ),
        usage_id=usage_id,
        tenant_id=event["tenant_id"],
        gcid=event["gcid"],
        model_id=event["model_id"],
        input_tokens=event["input_tokens"],
        output_tokens=event["output_tokens"],
        cached_tokens=event["cached_tokens"],
        cost_micros=event["cost_micros"],
        invocation_id=event["invocation_id"],
        agent_role=event.get("agent_role") or "",
        recorded_at=_timestamp(occurred_at),
        vendor=event.get("vendor") or "",
        fallback_chain=list(event.get("fallback_chain") or []),
        model_armor_verdict_pre=int(event.get("armor_pre") or 0),
        model_armor_verdict_post=int(event.get("armor_post") or 0),
        gateway_version=event.get("gateway_version") or "",
        mana_units=int(event.get("mana_units") or 0),
        action_code=event.get("action_code") or "",
    )
    envelope = {
        "event_id": usage_id,
        "idempotency_key": usage_id,
        "tenant_id": event["tenant_id"],
        "gcid": event["gcid"],
        "traceparent": event.get("traceparent") or "",
        "tracestate": event.get("tracestate") or "",
        "occurred_at": occurred_at.isoformat(),
        "source_service": SOURCE_SERVICE,
        "schema_version": str(SCHEMA_VERSION),
    }
    return record.SerializeToString(), envelope


class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=20)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    def _pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise RuntimeError("database pool is not started")
        return self.pool

    async def ping(self) -> bool:
        async with self._pool().acquire() as connection:
            return await connection.fetchval("SELECT 1") == 1

    async def budget(self, tenant_id: str) -> dict[str, Any] | None:
        if not _is_uuid(tenant_id):
            raise ValueError(f"invalid tenant_id {tenant_id!r} (must be a UUID)")
        async with self._pool().acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('chora.tenant_id',$1,true)",
                tenant_id,
            )
            row = await connection.fetchrow(
                """
                SELECT budget_usd_micros, spent_usd_micros, policy,
                       downgrade_to_logical_model_id
                  FROM per_tenant_llm_budget
                 WHERE tenant_id=$1::uuid
                   AND budget_period_start<=now()
                   AND budget_period_end>now()
                 ORDER BY budget_period_start DESC
                 LIMIT 1 FOR UPDATE
                """,
                tenant_id,
            )
            return dict(row) if row else None

    async def claim(self, gcid: str, key: str, action: str, invocation_id: str) -> bool:
        """Take the keyed debit claim for (gcid, dispatch key, action).

        The first claim inserts the row carrying the invocation that won it and
        returns True; a redelivery inserts nothing, records the dedupe on the
        winning row, and returns False. Port of the original Go ClaimDebit
        against observability's 0019 schema."""
        if not key:
            return True
        if not gcid:
            raise ValueError("gcid required")
        if not _is_uuid(gcid):
            raise ValueError(f"invalid gcid {gcid!r} (must be a UUID)")
        if not invocation_id:
            raise ValueError("invocation_id required")

        action = action or "unspecified"
        async with self._pool().acquire() as connection, connection.transaction():
            tag = await connection.execute(
                """
                INSERT INTO invoke_debit_claims
                    (gcid, dispatch_idempotency_key, action_code, invocation_id)
                VALUES($1::uuid,$2,$3,$4)
                ON CONFLICT (gcid, dispatch_idempotency_key, action_code) DO NOTHING
                """,
                gcid,
                key,
                action,
                invocation_id,
            )
            if tag.endswith("1"):
                return True

            await connection.execute(
                """
                UPDATE invoke_debit_claims
                   SET dedupe_count=dedupe_count+1,
                       last_deduped_invocation_id=$4,
                       last_deduped_at=now()
                 WHERE gcid=$1::uuid
                   AND dispatch_idempotency_key=$2
                   AND action_code=$3
                """,
                gcid,
                key,
                action,
                invocation_id,
            )
            return False

    async def debit_and_enqueue(self, event: dict[str, Any], debit: int) -> None:
        """Debit the budget and enqueue the TokenUsageRecorded outbox row in ONE
        transaction (transactional outbox).

        The gateway no longer writes token_usage_ledger — observability owns the
        ledger and populates it from this event. `usage_id` (= invocation_id) is
        the outbox idempotency key, so a redelivery inserts nothing; the debit is
        gated on that insert, so a redelivery cannot debit the budget twice."""
        occurred_at = _as_utc(event.get("occurred_at") or datetime.now(UTC))
        payload, envelope = _build_outbox(event, occurred_at)
        async with self._pool().acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('chora.tenant_id',$1,true)",
                event["tenant_id"],
            )
            # outbox_events RLS keys off app.current_tenant (not chora.tenant_id),
            # so set both: budget RLS uses chora.tenant_id, outbox RLS uses this.
            await connection.execute(
                "SELECT set_config('app.current_tenant',$1,true)",
                event["tenant_id"],
            )
            inserted = await connection.fetchval(
                """
                INSERT INTO outbox_events(
                    id,tenant_id,gcid,agid,aggregate_type,aggregate_id,event_type,topic,
                    payload,envelope,occurred_at,status,retry_count,idempotency_key)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,'pending',0,$12)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING id
                """,
                event["usage_id"],
                event["tenant_id"],
                event["gcid"],
                event.get("agent_id") or "",
                "token_usage",
                event["usage_id"],
                TOKEN_USAGE_EVENT_TYPE,
                TOKEN_USAGE_TOPIC,
                payload,
                json.dumps(envelope),
                occurred_at,
                event["usage_id"],
            )
            if inserted is not None and debit > 0:
                await connection.execute(
                    """
                    UPDATE per_tenant_llm_budget
                       SET spent_usd_micros=spent_usd_micros+$2,updated_at=now()
                     WHERE tenant_id=$1::uuid
                       AND budget_period_start<=now()
                       AND budget_period_end>now()
                    """,
                    event["tenant_id"],
                    debit,
                )
