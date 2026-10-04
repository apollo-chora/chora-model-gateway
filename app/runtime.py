from __future__ import annotations
import asyncio,base64,json
from dataclasses import dataclass,field
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.models.openai_responses import OpenAIResponsesModel
@dataclass(slots=True)
class Result:
 text:str="";model:str="";vendor:str="";input_tokens:int=0;output_tokens:int=0;cached_tokens:int=0;tool_calls:list=field(default_factory=list);citations:list=field(default_factory=list);search_queries:list=field(default_factory=list);image:bytes=b"";mime:str="";revised_prompt:str="";finish:str="stop"
class Runtime:
 def __init__(self,search_provider="exa"):self.search_provider=search_provider
 def args(self,s):
  d={"api_key":s.api_key or "not-used"}
  if s.base_url:d["base_url"]=s.base_url
  if s.extra_headers:d["default_headers"]=s.extra_headers
  return d
 async def generate(self,s,prompt,system="",messages=None,tools=None,params=None,grounded=False):
  p=dict(params or {})
  if s.format=="messages":return await self.anthropic(s,prompt,system,messages,tools,p,grounded)
  m=OpenAIResponsesModel(model_id=s.model,client_args=self.args(s),params=p|({"tools":[{"type":"web_search"}]} if grounded and s.support_grounding else {})) if s.format=="responses" else OpenAIModel(model_id=s.model,client_args=self.args(s),params=p)
  a=Agent(model=m,system_prompt=system or None,builtin_tools=({"web_search":self.search_provider} if grounded and not s.support_grounding else None))
  r=await asyncio.to_thread(a,prompt); return Result(text=str(r),model=s.model,vendor=s.format)
 async def anthropic(self,s,prompt,system,messages,tools,p,grounded):
  c=AsyncAnthropic(api_key=s.api_key,base_url=s.base_url or None,default_headers=s.extra_headers or None); ts=list(tools or [])
  if grounded:ts.append({"type":"web_search_20250305","name":"web_search","max_uses":5})
  kw={"model":s.model,"messages":messages or [{"role":"user","content":prompt}],"max_tokens":int(p.get("max_tokens",s.max_output_tokens or 4096))}
  if system:kw["system"]=system
  if ts:kw["tools"]=ts
  r=await c.messages.create(**kw); text="".join(getattr(x,"text","") for x in r.content); tc=[{"id":x.id,"type":"function","function":{"name":x.name,"arguments":json.dumps(x.input)}} for x in r.content if getattr(x,"type","")=="tool_use"]; return Result(text,r.model,"messages",r.usage.input_tokens,r.usage.output_tokens,tool_calls=tc,finish=r.stop_reason or "stop")
 async def image(self,s,prompt,p):
  r=await AsyncOpenAI(**self.args(s)).images.generate(model=s.model,prompt=prompt,**{k:v for k,v in p.items() if k in {"size","quality","style"}}); d=r.data[0]; raw=base64.b64decode(d.b64_json); return Result(model=s.model,vendor="images",image=raw,mime="image/webp" if raw[:4]==b"RIFF" else "image/png",revised_prompt=getattr(d,"revised_prompt","") or "")
 async def embed(self,s,text,dims=0):
  kw={"model":s.model,"input":text}
  if dims:kw["dimensions"]=dims
  r=await AsyncOpenAI(**self.args(s)).embeddings.create(**kw); return r.data[0].embedding,getattr(r.usage,"prompt_tokens",0),r.model
