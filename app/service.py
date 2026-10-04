import time,uuid
from .runtime import Runtime
class GatewayError(Exception):
 def __init__(self,status,code,message):super().__init__(message);self.status,self.code=status,code
class Gateway:
 def __init__(self,settings,models,db):self.settings,self.models,self.db,self.runtime=settings,models,db,Runtime(settings.search_provider)
 def model(self,n):
  s=self.models.get((n or "").lower())
  if not s:raise GatewayError(404,"model_not_found",f'model "{n}" is not in the registry')
  return s
 def cost(self,s,i,o,c=0):
  p=s.pricing;return(i*int(p.get("input_per_mtok_usd_micros",0))+o*int(p.get("output_per_mtok_usd_micros",0))+c*int(p.get("cached_per_mtok_usd_micros",0)))//1000000
 async def invoke(self,q):
  st=time.monotonic();iid=q.get("invocation_id") or str(uuid.uuid4());s=self.model(q.get("model") or q.get("logical_model_id"));t=q.get("tenant_id") or self.settings.default_tenant_id;g=q.get("gcid") or self.settings.default_gcid;a=q.get("agent_id") or self.settings.default_agent_id;b=await self.db.budget(t)
  if b and b["spent_usd_micros"]>=b["budget_usd_micros"]:
   if b["policy"]=="block":raise GatewayError(402,"budget_exhausted","tenant LLM budget exhausted")
   if b["policy"]=="downgrade" and b["downgrade_to_logical_model_id"]:s=self.model(b["downgrade_to_logical_model_id"])
  claimed=await self.db.claim(g,q.get("dispatch_idempotency_key",""),q.get("action_code",a));chain=[s]+[self.model(x) for x in q.get("fallback_ids",[]) if x.lower() in self.models];z=None;last=None;tried=[]
  for x in chain:
   tried.append(f"{x.format}:{x.model}")
   try:z=await(self.runtime.image(x,q.get("prompt",""),q.get("params",{})) if q.get("modality")=="IMAGE" else self.runtime.generate(x,q.get("prompt",""),q.get("system",""),q.get("messages"),q.get("tools"),q.get("params",{}),q.get("modality")=="GROUNDED"));s=x;break
   except Exception as e:last=e
  if z is None:raise GatewayError(502,"vendor_error",f"provider chain exhausted: {last}")
  cost=self.cost(s,z.input_tokens,z.output_tokens,z.cached_tokens) if claimed else 0;e={"invocation_id":iid,"tenant_id":t,"gcid":g,"model_id":z.model or s.model,"vendor":z.vendor,"input_tokens":z.input_tokens,"output_tokens":z.output_tokens,"cached_tokens":z.cached_tokens,"agent_role":q.get("action_code") or a,"surface":q.get("surface"),"modality":q.get("modality") or "TEXT","fallback_chain":tried,"debit_deduped":not claimed,"traceparent":q.get("traceparent"),"tracestate":q.get("tracestate"),"gateway_version":self.settings.service_version};await self.db.settle(e,cost);return{"id":iid,"result":z,"model":z.model or s.model,"vendor":z.vendor,"usage":{"input":z.input_tokens,"output":z.output_tokens,"cached":z.cached_tokens,"cost":cost},"fallback_chain":tried,"latency_ms":int((time.monotonic()-st)*1000)}
 async def embed(self,q):
  iid=q.get("invocation_id") or str(uuid.uuid4());s=self.model(q.get("model") or q.get("logical_model_id") or "text-embedding-004");t=q.get("tenant_id") or self.settings.default_tenant_id;g=q.get("gcid") or self.settings.default_gcid;a=q.get("agent_id") or self.settings.default_agent_id;v,n,m=await self.runtime.embed(s,q["text"],int(q.get("dimensions",0)));await self.db.settle({"invocation_id":iid,"tenant_id":t,"gcid":g,"model_id":m,"vendor":"embeddings","input_tokens":n,"output_tokens":0,"cached_tokens":0,"agent_role":a,"modality":"TEXT","fallback_chain":[f"embeddings:{m}"],"traceparent":q.get("traceparent"),"tracestate":q.get("tracestate"),"gateway_version":self.settings.service_version},0);return{"id":iid,"values":v,"model":m,"vendor":"embeddings","input_tokens":n}
