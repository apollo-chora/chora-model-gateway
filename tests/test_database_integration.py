"""PostgreSQL integration tests for the budget/claim/outbox ports.

These run against a real PostgreSQL and are skipped unless
``CHORA_TEST_DATABASE_URL`` is set. The database MUST be initialised from
chora-observability's migrations — the authoritative schema — not from
gateway-owned copies, because the gateway writes observability's tables
(``per_tenant_llm_budget``, ``invoke_debit_claims``, ``outbox_events``) and no
longer owns ``token_usage_ledger``.

The application role (chora_app) holds no DELETE privilege and RLS applies on
``per_tenant_llm_budget``, so the fixture creates rows under the tenant GUC and
cleans up through the superuser DSN in ``CHORA_TEST_ADMIN_DATABASE_URL`` when
it is available.
"""

from __future__ import annotations

import os
import uuid

import pytest
from chora_contracts_gen.events.observability import token_usage_pb2

from app.database import Database

pytestmark = pytest.mark.skipif(
    not os.environ.get("CHORA_TEST_DATABASE_URL"),
    reason="CHORA_TEST_DATABASE_URL is not set",
)

GCID = "00000000-0000-7000-8000-000000000002"
TOPIC = "chora.observability.token_usage.recorded.v1"


@pytest.fixture
async def db():
    database = Database(os.environ["CHORA_TEST_DATABASE_URL"])
    await database.start()
    yield database
    await database.close()


@pytest.fixture
async def admin():
    """A superuser connection for row cleanup (chora_app cannot DELETE)."""
    url = os.environ.get("CHORA_TEST_ADMIN_DATABASE_URL")
    if not url:
        yield None
        return
    database = Database(url)
    await database.start()
    yield database
    await database.close()


@pytest.fixture
async def tenant(db, admin):
    """An isolated tenant with a budget row, created and cleaned up per test."""
    tenant_id = str(uuid.uuid4())
    async with db._pool().acquire() as connection, connection.transaction():
        await connection.execute("SELECT set_config('chora.tenant_id', $1, true)", tenant_id)
        await connection.execute(
            """
            INSERT INTO per_tenant_llm_budget
                (tenant_id, budget_period_start, budget_period_end,
                 budget_usd_micros, spent_usd_micros, policy)
            VALUES ($1::uuid, now(), now() + INTERVAL '30 days', 1000000, 0, 'block')
            """,
            tenant_id,
        )
    yield tenant_id
    if admin is not None:
        async with admin._pool().acquire() as connection:
            await connection.execute("DELETE FROM outbox_events WHERE envelope->>'tenant_id'=$1", tenant_id)
            await connection.execute("DELETE FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid", tenant_id)


def _event(tenant: str, invocation_id: str, cost: int = 165) -> dict:
    return {
        "usage_id": invocation_id,
        "invocation_id": invocation_id,
        "tenant_id": tenant,
        "gcid": GCID,
        "model_id": "muse-spark-1.3-contributor",
        "vendor": "openai",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 10,
        "cost_micros": cost,
        "agent_role": "openai_compat",
        "action_code": "agent",
        "fallback_chain": ["openai:muse-spark-1.3-contributor"],
        "traceparent": "00-abc",
        "tracestate": "foo=bar",
        "gateway_version": "test",
    }


async def _as_tenant(db, tenant_id, fn):
    """Run a read under the tenant's RLS scope, in one transaction."""
    async with db._pool().acquire() as connection, connection.transaction():
        await connection.execute("SELECT set_config('chora.tenant_id', $1, true)", tenant_id)
        return await fn(connection)


@pytest.mark.anyio
async def test_ping(db):
    assert await db.ping() is True


@pytest.mark.anyio
async def test_no_budget_row_allows(db, tenant, admin):
    assert await db.budget(tenant) is not None  # the fixture created one
    async with admin._pool().acquire() as connection:
        await connection.execute("DELETE FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid", tenant)
    assert await db.budget(tenant) is None


@pytest.mark.anyio
async def test_claim_records_the_winning_invocation_and_dedupes(db):
    key = f"key-{uuid.uuid4()}"
    winner = str(uuid.uuid4())
    loser = str(uuid.uuid4())
    assert await db.claim(GCID, key, "agent", winner) is True
    assert await db.claim(GCID, key, "agent", loser) is False
    # A different action code is a different claim.
    assert await db.claim(GCID, key, "other", winner) is True

    async with db._pool().acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT invocation_id, dedupe_count, last_deduped_invocation_id
              FROM invoke_debit_claims
             WHERE gcid=$1::uuid AND dispatch_idempotency_key=$2 AND action_code=$3
            """,
            GCID,
            key,
            "agent",
        )
    assert row["invocation_id"] == winner
    assert row["dedupe_count"] == 1
    assert row["last_deduped_invocation_id"] == loser


@pytest.mark.anyio
async def test_claim_requires_a_uuid_gcid(db):
    with pytest.raises(ValueError):
        await db.claim("not-a-uuid", "key", "agent", str(uuid.uuid4()))


@pytest.mark.anyio
async def test_claim_requires_an_invocation_id(db):
    with pytest.raises(ValueError):
        await db.claim(GCID, "key", "agent", "")


@pytest.mark.anyio
async def test_debit_and_enqueue_debits_and_enqueues_atomically(db, tenant, admin):
    invocation_id = str(uuid.uuid4())
    await db.debit_and_enqueue(_event(tenant, invocation_id, cost=165), 165)

    async def read(connection):
        spent = await connection.fetchval(
            "SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid",
            tenant,
        )
        row = await connection.fetchrow(
            """
            SELECT payload, envelope, event_type, topic, status, idempotency_key, aggregate_type
              FROM outbox_events WHERE id=$1
            """,
            invocation_id,
        )
        return spent, row

    spent, row = await _as_tenant(db, tenant, read)
    assert spent == 165
    assert row is not None
    assert row["status"] == "pending"
    assert row["event_type"] == TOPIC
    assert row["topic"] == TOPIC
    assert row["aggregate_type"] == "token_usage"
    assert row["idempotency_key"] == invocation_id

    # The payload round-trips as the canonical TokenUsageRecorded proto.
    record = token_usage_pb2.TokenUsageRecorded()
    record.ParseFromString(bytes(row["payload"]))
    assert record.invocation_id == invocation_id
    assert record.usage_id == invocation_id
    assert record.tenant_id == tenant
    assert record.cost_micros == 165
    assert record.input_tokens == 100
    assert record.vendor == "openai"
    assert record.envelope.tenant_id == tenant
    assert record.envelope.source_service == "chora-model-gateway"

    # The gateway must NOT write observability's ledger.
    if admin is not None:
        async with admin._pool().acquire() as connection:
            ledger_rows = await connection.fetchval(
                "SELECT count(*) FROM token_usage_ledger WHERE tenant_id=$1::uuid", tenant
            )
        assert ledger_rows == 0


@pytest.mark.anyio
async def test_debit_and_enqueue_zero_debit_still_enqueues(db, tenant):
    invocation_id = str(uuid.uuid4())
    await db.debit_and_enqueue(_event(tenant, invocation_id, cost=0), 0)

    async def read(connection):
        spent = await connection.fetchval(
            "SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid",
            tenant,
        )
        count = await connection.fetchval("SELECT count(*) FROM outbox_events WHERE id=$1", invocation_id)
        return spent, count

    spent, count = await _as_tenant(db, tenant, read)
    assert spent == 0
    assert count == 1


@pytest.mark.anyio
async def test_debit_and_enqueue_is_idempotent_on_the_budget(db, tenant):
    """A redelivered settlement must not debit twice or duplicate the event."""
    invocation_id = str(uuid.uuid4())
    await db.debit_and_enqueue(_event(tenant, invocation_id, cost=165), 165)
    await db.debit_and_enqueue(_event(tenant, invocation_id, cost=165), 165)

    async def read(connection):
        spent = await connection.fetchval(
            "SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid",
            tenant,
        )
        count = await connection.fetchval("SELECT count(*) FROM outbox_events WHERE id=$1", invocation_id)
        return spent, count

    spent, count = await _as_tenant(db, tenant, read)
    assert spent == 165  # not 330 — the redelivery must not debit again
    assert count == 1


@pytest.mark.anyio
async def test_rls_scopes_budget_rows_to_the_tenant(db, tenant, admin):
    other = str(uuid.uuid4())
    async with admin._pool().acquire() as connection:
        await connection.execute(
            """
            INSERT INTO per_tenant_llm_budget
                (tenant_id, budget_period_start, budget_period_end,
                 budget_usd_micros, spent_usd_micros, policy)
            VALUES ($1::uuid, now(), now() + INTERVAL '30 days', 5, 0, 'block')
            """,
            other,
        )

    async def read(connection):
        return await connection.fetchval("SELECT count(*) FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid", other)

    try:
        # Reading another tenant's row under this tenant's GUC returns nothing.
        assert await _as_tenant(db, tenant, read) == 0
    finally:
        async with admin._pool().acquire() as connection:
            await connection.execute("DELETE FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid", other)
