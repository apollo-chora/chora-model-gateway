import json,asyncpg
class Database:
 def __init__(self,dsn):self.dsn,self.pool=dsn,None
 async def start(self):self.pool=await asyncpg.create_pool(self.dsn,min_size=1,max_size=20)
 async def close(self):
  if self.pool:await self.pool.close()
 async def ping(self):
  async with self.pool.acquire() as c:return await c.fetchval("SELECT 1")==1
 async def budget(self,t):
  async with self.pool.acquire() as c:
   async with c.transaction():
    await c.execute("SELECT set_config('chora.tenant_id',$1,true)",t); r=await c.fetchrow("SELECT budget_usd_micros,spent_usd_micros,policy,downgrade_to_logical_model_id FROM per_tenant_llm_budget WHERE tenant_id=$1::uuid AND budget_period_start<=now() AND budget_period_end>now() ORDER BY budget_period_start DESC LIMIT 1 FOR UPDATE",t); return dict(r) if r else None
 async def claim(self,g,k,a):
  if not k:return True
  a=a or "unspecified"
  async with self.pool.acquire() as c:
   async with c.transaction():
    tag=await c.execute("INSERT INTO invoke_debit_claims(gcid,dispatch_idempotency_key,action_code) VALUES($1::uuid,$2,$3) ON CONFLICT DO NOTHING",g,k,a)
    if tag.endswith("1"):return True
    await c.execute("UPDATE invoke_debit_claims SET dedupe_count=dedupe_count+1,last_deduped_at=now() WHERE gcid=$1::uuid AND dispatch_idempotency_key=$2 AND action_code=$3",g,k,a); return False
 async def settle(self,e,cost):
  async with self.pool.acquire() as c:
   async with c.transaction():
    await c.execute("SELECT set_config('chora.tenant_id',$1,true)",e["tenant_id"])
    if cost>0:await c.execute("UPDATE per_tenant_llm_budget SET spent_usd_micros=spent_usd_micros+$2,updated_at=now() WHERE tenant_id=$1::uuid AND budget_period_start<=now() AND budget_period_end>now()",e["tenant_id"],cost)
    await c.execute("INSERT INTO token_usage_ledger(invocation_id,tenant_id,gcid,model_id,vendor,input_tokens,output_tokens,cached_tokens,cost_usd_micros,agent_role,surface,modality,fallback_chain,debit_deduped,traceparent,tracestate,gateway_version,recorded_at) VALUES($1,$2::uuid,$3::uuid,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14,$15,$16,$17,now()) ON CONFLICT(invocation_id) DO NOTHING",e["invocation_id"],e["tenant_id"],e["gcid"],e["model_id"],e["vendor"],e["input_tokens"],e["output_tokens"],e["cached_tokens"],cost,e.get("agent_role"),e.get("surface") or "unspecified",e.get("modality") or "TEXT",json.dumps(e.get("fallback_chain",[])),e.get("debit_deduped",False),e.get("traceparent"),e.get("tracestate"),e.get("gateway_version"))
