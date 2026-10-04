import base64,time
from blacksheep import Application,Request
from blacksheep.server.responses import json as reply
from .service import GatewayError
def create_app(gw,db,models):
 app=Application()
 @app.exception_handler(GatewayError)
 async def err(_,e):return reply({"error":{"message":str(e),"type":"invalid_request_error" if e.status<500 else "upstream_error","code":e.code}},status=e.status)
 @app.router.get("/healthz")
 async def health():return reply({"status":"ok"})
 @app.router.get("/readyz")
 async def ready():
  ok=await db.ping();return reply({"status":"ready" if ok else "not_ready"},status=200 if ok else 503)
 @app.router.get("/v1/models")
 async def lm():return reply({"object":"list","data":[{"id":k,"object":"model","created":0,"owned_by":"chora"} for k in dict.fromkeys(models)]})
 @app.router.get("/v1/models/{model}")
 async def gm(model:str):gw.model(model);return reply({"id":model,"object":"model","created":0,"owned_by":"chora"})
 def attr(r):
  def h(n):
   x=r.get_first_header(n);return x.decode() if x else ""
  return{"tenant_id":h(b"x-chora-tenant-id") or None,"gcid":h(b"x-chora-gcid") or None,"traceparent":h(b"traceparent"),"tracestate":h(b"tracestate")}
 @app.router.post("/v1/chat/completions")
 @app.router.post("/v1/completions")
 async def chat(r:Request):
  b=await r.json()
  if b.get("stream"):raise GatewayError(501,"streaming_unsupported",'this gateway does not implement SSE streaming; retry with "stream": false')
  ms=b.get("messages") or[{"role":"user","content":b.get("prompt","")}];p="";s=""
  for m in ms:
   if m.get("role")=="system":s+=str(m.get("content",""))
   elif m.get("role")=="user":p=str(m.get("content",""))
  tools=b.get("tools",[]);ground=any("web_search" in str(x.get("type","")).lower() for x in tools);x=await gw.invoke(attr(r)|{"model":b.get("model"),"prompt":p,"system":s,"messages":ms,"tools":tools,"params":{k:b[k] for k in("temperature","top_p","max_tokens","seed","stop") if k in b},"modality":"GROUNDED" if ground else "TEXT","action_code":b.get("user"),"surface":"openai_compat"});z=x["result"];msg={"role":"assistant","content":z.text}
  if z.tool_calls:msg["tool_calls"]=z.tool_calls
  return reply({"id":x["id"],"object":"chat.completion","created":int(time.time()),"model":x["model"],"choices":[{"index":0,"message":msg,"finish_reason":"tool_calls" if z.tool_calls else "stop"}],"usage":{"prompt_tokens":x["usage"]["input"],"completion_tokens":x["usage"]["output"],"total_tokens":x["usage"]["input"]+x["usage"]["output"]}})
 @app.router.post("/v1/responses")
 async def responses(r:Request):
  b=await r.json()
  if b.get("stream"):raise GatewayError(501,"streaming_unsupported",'this gateway does not implement SSE streaming; retry with "stream": false')
  ground=any("web_search" in str(x.get("type","")).lower() or "web_fetch" in str(x.get("type","")).lower() for x in b.get("tools",[]));inp=b.get("input","");p=inp if isinstance(inp,str) else "\n".join(str(x.get("content",x.get("text",""))) for x in inp);x=await gw.invoke(attr(r)|{"model":b.get("model"),"prompt":p,"system":b.get("instructions",""),"params":{k:b[k] for k in("temperature","top_p","max_output_tokens","max_tokens","seed","reasoning_effort") if k in b},"modality":"GROUNDED" if ground else "TEXT","action_code":b.get("user"),"surface":"responses"});z=x["result"];out=[]
  if ground:out.append({"type":"web_search_call","id":"ws_"+x["id"],"status":"completed"})
  out.append({"type":"message","id":"msg_"+x["id"],"status":"completed","role":"assistant","content":[{"type":"output_text","text":z.text,"annotations":z.citations}]});return reply({"id":x["id"],"object":"response","created_at":int(time.time()),"model":x["model"],"status":"completed","output":out,"usage":{"prompt_tokens":x["usage"]["input"],"completion_tokens":x["usage"]["output"],"total_tokens":x["usage"]["input"]+x["usage"]["output"]},"chora_gateway":{"vendor":x["vendor"],"grounded":ground,"citations":z.citations,"search_queries":z.search_queries,"fallback_chain":x["fallback_chain"],"latency_ms":x["latency_ms"],"invocation_id":x["id"],"grounding_surface":"responses" if ground else ""}})
 @app.router.post("/v1/images/generations")
 @app.router.post("/v1/images/edits")
 async def image(r:Request):
  b=await r.json()
  if b.get("response_format")=="url":raise GatewayError(501,"url_response_unsupported","this gateway returns bytes, not hosted URLs")
  x=await gw.invoke(attr(r)|{"model":b.get("model"),"prompt":b.get("prompt",""),"params":{k:b[k] for k in("size","quality","style") if k in b},"modality":"IMAGE","action_code":b.get("user"),"surface":"openai_compat"});z=x["result"];return reply({"created":int(time.time()),"data":[{"b64_json":base64.b64encode(z.image).decode(),"revised_prompt":z.revised_prompt}],"usage":{"prompt_tokens":x["usage"]["input"],"completion_tokens":x["usage"]["output"],"total_tokens":x["usage"]["input"]+x["usage"]["output"]},"mime_type":z.mime})
 @app.router.post("/v1/embeddings")
 async def embeddings(r:Request):
  b=await r.json();xs=b.get("input");xs=[xs] if isinstance(xs,str) else xs
  if not isinstance(xs,list) or not xs:raise GatewayError(400,"invalid_parameter",'"input" must be a string or an array of strings')
  rows=[];total=0
  for i,t in enumerate(xs):
   e=await gw.embed(attr(r)|{"model":b.get("model"),"text":t,"dimensions":int(b.get("dimensions",0)),"agent_id":gw.settings.default_agent_id});rows.append({"object":"embedding","index":i,"embedding":e["values"]});total+=e["input_tokens"]
  return reply({"object":"list","model":b.get("model"),"data":rows,"usage":{"prompt_tokens":total,"total_tokens":total}})
 return app
