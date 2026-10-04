"""PostgreSQL integration tests for the budget/claim/ledger ports.

These run against a real PostgreSQL and are skipped unless
``CHORA_TEST_DATABASE_URL`` is set. The Docker verification sets it to the
Compose database, which has already applied migrations/0001-0003.

The application role (chora_app) holds no DELETE privilege and RLS applies,
so the fixture creates rows under the tenant GUC and cleans up through the
superuser DSN in ``CHORA_TEST_ADMIN_DATABASE_URL`` when it is available.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.database import Database

pytestmark = pytest.mark.skipif(
    not os.environ.get("CHORA_TEST_DATABASE_URL"),
    reason="CHORA_TEST_DATABASE_URL is not set",
)

GCID = "00000000-0000-7000-8000-000000000002"


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
    """An isolated tenant with a budget row, created and cleaned up per test.

    The INSERT runs under the tenant GUC inside one transaction (the GUC is
    transaction-local); cleanup uses the superuser DSN because chora_app
    holds no DELETE privilege.
    """
    tenant_id = str(uuid.uuid4())
    async with db._pool().acquire() as connection, connection.transaction():
        await connection.execute("SELECT set_config('chora.tenant_id', $1, true)", tenant_id)
        await connection.execute(
            """
            INSERT INTO per_tenant_llm_budget
                (tenant_id, budget_usd_micros, spent_usd_micros, policy)
            VALUES ($1::uuid, 1000000, 0, 'block')
            """,
            tenant_id,
        )
    yield tenant_id
    if admin is not None:
        async with admin._pool().acquire() as connection:
            await connection.execute("DELETE FROM token_usage_ledger WHERE tenant_id=$1::uuid", tenant_id)
            await connection.execute("DELETE FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid", tenant_id)


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
async def test_claim_dedupes_on_the_key(db):
    key = f"key-{uuid.uuid4()}"
    assert await db.claim(GCID, key, "agent") is True
    assert await db.claim(GCID, key, "agent") is False
    # A different action code is a different claim.
    assert await db.claim(GCID, key, "other") is True


@pytest.mark.anyio
async def test_claim_requires_a_uuid_gcid(db):
    with pytest.raises(ValueError):
        await db.claim("not-a-uuid", "key", "agent")


@pytest.mark.anyio
async def test_settle_writes_the_debit_and_the_row_atomically(db, tenant):
    event = {
        "invocation_id": str(uuid.uuid4()),
        "tenant_id": tenant,
        "gcid": GCID,
        "model_id": "muse-spark-1.3-contributor",
        "vendor": "openai",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 10,
        "agent_role": "openai_compat",
        "surface": "openai_compat",
        "modality": "TEXT",
        "fallback_chain": ["openai:muse-spark-1.3-contributor"],
        "debit_deduped": False,
        "traceparent": None,
        "tracestate": None,
        "gateway_version": "test",
        "cost_usd_micros": 165,
    }
    await db.settle(event, 165)

    async def read(connection):
        spent = await connection.fetchval(
            "SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid",
            tenant,
        )
        row = await connection.fetchrow(
            "SELECT cost_usd_micros, input_tokens, debit_deduped FROM token_usage_ledger WHERE invocation_id=$1",
            event["invocation_id"],
        )
        return spent, row

    spent, row = await _as_tenant(db, tenant, read)
    assert spent == 165
    assert row["cost_usd_micros"] == 165
    assert row["input_tokens"] == 100
    assert row["debit_deduped"] is False


@pytest.mark.anyio
async def test_settle_with_zero_debit_still_ledgers(db, tenant):
    event = {
        "invocation_id": str(uuid.uuid4()),
        "tenant_id": tenant,
        "gcid": GCID,
        "model_id": "m",
        "vendor": "openai",
        "input_tokens": 10,
        "output_tokens": 0,
        "cached_tokens": 0,
        "agent_role": "a",
        "surface": "s",
        "modality": "TEXT",
        "fallback_chain": [],
        "debit_deduped": True,
        "traceparent": None,
        "tracestate": None,
        "gateway_version": "test",
        "cost_usd_micros": 0,
    }
    await db.settle(event, 0)

    async def read(connection):
        spent = await connection.fetchval(
            "SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid",
            tenant,
        )
        count = await connection.fetchval(
            "SELECT count(*) FROM token_usage_ledger WHERE invocation_id=$1",
            event["invocation_id"],
        )
        return spent, count

    spent, count = await _as_tenant(db, tenant, read)
    assert spent == 0
    assert count == 1


@pytest.mark.anyio
async def test_rls_scopes_rows_to_the_tenant(db, tenant):
    other = str(uuid.uuid4())
    event = {
        "invocation_id": str(uuid.uuid4()),
        "tenant_id": other,
        "gcid": GCID,
        "model_id": "m",
        "vendor": "openai",
        "input_tokens": 1,
        "output_tokens": 0,
        "cached_tokens": 0,
        "agent_role": "a",
        "surface": "s",
        "modality": "TEXT",
        "fallback_chain": [],
        "debit_deduped": False,
        "traceparent": None,
        "tracestate": None,
        "gateway_version": "test",
        "cost_usd_micros": 0,
    }
    # A ledger row for another tenant is written under that tenant's RLS scope.
    await db.settle(event, 0)

    # Reading it back under a different tenant's GUC returns nothing.
    async def read(connection):
        return await connection.fetchval(
            "SELECT count(*) FROM token_usage_ledger WHERE invocation_id=$1",
            event["invocation_id"],
        )

    visible = await _as_tenant(db, tenant, read)
    assert visible == 0
