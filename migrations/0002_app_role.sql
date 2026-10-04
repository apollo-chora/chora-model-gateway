-- 0002_app_role.sql — a NON-superuser application role.
--
-- WHY THIS FILE EXISTS
--
-- The local stack creates `chora` as POSTGRES_USER, which makes it a
-- SUPERUSER. PostgreSQL exempts superusers from row-level security entirely,
-- regardless of FORCE ROW LEVEL SECURITY. So an application connecting as
-- `chora` sees every tenant's rows and the RLS policies in 0001 are decorative
-- — verified directly: a SELECT with no `chora.tenant_id` GUC set returned
-- every budget row.
--
-- That is fine for psql and terrible for the gateway. This role is what the
-- gateway connects as: an ordinary role with table privileges and no
-- BYPASSRLS, so the policies actually bite.
--
-- Credentials are local-development only. A real deployment supplies its own
-- role and password out of band.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'chora_app') THEN
        CREATE ROLE chora_app LOGIN PASSWORD 'chora_app';
    END IF;
END
$$;

-- The gateway is not the owner of these tables and must never be able to
-- alter them, drop them, or read them across tenants.
GRANT USAGE ON SCHEMA public TO chora_app;
GRANT SELECT, INSERT, UPDATE ON per_tenant_llm_budget TO chora_app;
GRANT SELECT, INSERT         ON token_usage_ledger     TO chora_app;
GRANT SELECT, INSERT, UPDATE ON invoke_debit_claims    TO chora_app;

-- Sequence privileges, in case any table later grows a serial column.
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO chora_app;

-- No membership GRANT here on purpose.
--
-- An earlier version did `GRANT chora_app TO chora` as a convenience so a human
-- could `psql -U chora_app` and SET the tenant GUC. That is
-- deployment-specific — the local compose stack has a `chora` superuser, but a
-- server provisioned with `postgres` does not — and an unguarded GRANT against
-- a missing role ABORTS the migration, leaving the privileges above applied but
-- the file half-run. Observed live on 2026-10-04.
--
-- To inspect the ledger, just connect as chora_app with its password and set
-- the GUC in the same session:
--
--   psql "postgres://chora_app:<password>@host:5432/chora_observability" \
--     -c "BEGIN; SET LOCAL chora.tenant_id='<uuid>'; SELECT ...; COMMIT;"
--
-- Membership is only needed for passwordless admin access, which is an
-- operator decision, not a schema one:
--
--   GRANT chora_app TO <your-admin-role>;

-- Belt and braces: assert the exemption is actually gone. If a future change
-- grants BYPASSRLS to this role, this notice fires at migration time rather
-- than as a silent cross-tenant leak in production.
DO $$
DECLARE
    bypasses boolean;
BEGIN
    SELECT r.rolbypassrls INTO bypasses FROM pg_roles r WHERE r.rolname = 'chora_app';
    IF bypasses THEN
        RAISE EXCEPTION 'chora_app has BYPASSRLS — the tenant isolation in 0001 is not being enforced';
    END IF;
END
$$;