-- 0003_claim_rls.sql — let the application role write idempotency claims.
--
-- WHY THIS FILE EXISTS
--
-- 0001 enables FORCE ROW LEVEL SECURITY on invoke_debit_claims but declares
-- no policy for it. With FORCE RLS and no policy, a non-owner role that
-- does not bypass RLS (chora_app) can neither read nor insert a row — so
-- every keyed-idempotency claim failed with "new row violates row-level
-- security policy", and a redelivered dispatch debited the budget twice.
--
-- The table is keyed by gcid, not by tenant_id, so the tenant-isolation
-- policy used by the other two tables does not apply. The correct fix is a
-- permissive policy scoped to the application role: the claim row carries
-- no tenant data, and the gateway is the only writer.

DROP POLICY IF EXISTS invoke_debit_claims_app_access ON invoke_debit_claims;
CREATE POLICY invoke_debit_claims_app_access ON invoke_debit_claims
    TO chora_app
    USING (true)
    WITH CHECK (true);
