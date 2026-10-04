from __future__ import annotations

import json
from typing import Any

import asyncpg


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
        async with self._pool().acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT set_config('chora.tenant_id',$1,true)", tenant_id)
                row = await connection.fetchrow(
                    """SELECT budget_usd_micros, spent_usd_micros, policy,
                              downgrade_to_logical_model_id
                       FROM per_tenant_llm_budget
                      WHERE tenant_id=$1::uuid
                        AND budget_period_start<=now()
                        AND budget_period_end>now()
                      ORDER BY budget_period_start DESC
                      LIMIT 1 FOR UPDATE""",
                    tenant_id,
                )
                return dict(row) if row else None

    async def claim(self, gcid: str, key: str, action: str) -> bool:
        if not key:
            return True
        action = action or "unspecified"
        async with self._pool().acquire() as connection:
            async with connection.transaction():
                tag = await connection.execute(
                    """INSERT INTO invoke_debit_claims
                           (gcid, dispatch_idempotency_key, action_code)
                       VALUES($1::uuid,$2,$3) ON CONFLICT DO NOTHING""",
                    gcid, key, action,
                )
                if tag.endswith("1"):
                    return True
                await connection.execute(
                    """UPDATE invoke_debit_claims
                          SET dedupe_count=dedupe_count+1,last_deduped_at=now()
                        WHERE gcid=$1::uuid AND dispatch_idempotency_key=$2
                          AND action_code=$3""",
                    gcid, key, action,
                )
                return False

    async def settle(self, event: dict[str, Any], cost: int) -> None:
        async with self._pool().acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT set_config('chora.tenant_id',$1,true)", event["tenant_id"]
                )
                if cost > 0:
                    await connection.execute(
                        """UPDATE per_tenant_llm_budget
                              SET spent_usd_micros=spent_usd_micros+$2,updated_at=now()
                            WHERE tenant_id=$1::uuid
                              AND budget_period_start<=now()
                              AND budget_period_end>now()""",
                        event["tenant_id"], cost,
                    )
                await connection.execute(
                    """INSERT INTO token_usage_ledger(
                           invocation_id,tenant_id,gcid,model_id,vendor,input_tokens,
                           output_tokens,cached_tokens,cost_usd_micros,agent_role,
                           surface,modality,fallback_chain,debit_deduped,traceparent,
                           tracestate,gateway_version,recorded_at)
                       VALUES($1,$2::uuid,$3::uuid,$4,$5,$6,$7,$8,$9,$10,$11,$12,
                              $13::jsonb,$14,$15,$16,$17,now())
                       ON CONFLICT(invocation_id) DO NOTHING""",
                    event["invocation_id"], event["tenant_id"], event["gcid"],
                    event["model_id"], event["vendor"], event["input_tokens"],
                    event["output_tokens"], event["cached_tokens"], cost,
                    event.get("agent_role"), event.get("surface") or "unspecified",
                    event.get("modality") or "TEXT",
                    json.dumps(event.get("fallback_chain", [])),
                    event.get("debit_deduped", False), event.get("traceparent"),
                    event.get("tracestate"), event.get("gateway_version"),
                )
