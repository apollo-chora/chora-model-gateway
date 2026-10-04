from __future__ import annotations
import os
from dataclasses import dataclass,field
from pathlib import Path
import yaml
@dataclass(slots=True)
class ModelSpec:
 id:str; kind:str="text"; format:str="chat_completions"; model:str=""; base_url:str=""; api_key_env:str=""; capabilities:list[str]=field(default_factory=lambda:["chat"]); fallback_ids:list[str]=field(default_factory=list); aliases:list[str]=field(default_factory=list); context_window:int=0; max_output_tokens:int=0; pricing:dict=field(default_factory=dict); extra_headers:dict[str,str]=field(default_factory=dict); support_grounding:bool=False
 @property
 def api_key(self): return os.getenv(self.api_key_env,"") if self.api_key_env else ""
@dataclass(slots=True)
class Settings:
 database_url:str; default_tenant_id:str; default_gcid:str; default_agent_id:str; http_port:int; grpc_port:int; registry_path:str; service_version:str; search_provider:str
 @classmethod
 def load(cls):
  t=os.environ["CHORA_DEFAULT_TENANT_ID"]; return cls(os.environ["CHORA_DATABASE_URL"],t,os.getenv("CHORA_DEFAULT_GCID",t),os.getenv("CHORA_DEFAULT_AGENT_ID","openai_compat"),int(os.getenv("CHORA_HTTP_PORT","8080")),int(os.getenv("CHORA_GRPC_PORT","9090")),os.getenv("CHORA_MODEL_REGISTRY","config/models.yaml"),os.getenv("SERVICE_VERSION","chora-model-gateway:local"),os.getenv("SEARCH_PROVIDER","exa"))
def _env(prefix,kind,fmt):
 m=os.getenv(f"{prefix}_LLM_MODEL","").strip()
 if not m:return None
 g=os.getenv(f"{prefix}_LLM_SUPPORT_GROUNDING","false").lower() in {"1","true","yes","on"}
 return ModelSpec(m,kind,os.getenv(f"{prefix}_LLM_FORMAT",fmt),m,os.getenv(f"{prefix}_LLM_BASE_URL",""),f"{prefix}_LLM_API_KEY",([kind] if kind!="text" else ["chat","tools"]+(["web_search"] if g else [])),support_grounding=g)
def load_models(path):
 specs=[]; p=Path(path)
 if p.exists():
  for r in (yaml.safe_load(p.read_text()) or {}).get("models",[]):
   caps=r.get("capabilities",["chat"]); provider=r.get("provider",""); fmt=r.get("format") or ("messages" if provider=="anthropic" else "chat_completions"); kind="image" if "image" in caps and "chat" not in caps else ("embedding" if "embeddings" in caps and "chat" not in caps else "text")
   specs.append(ModelSpec(r["id"],kind,fmt,r.get("upstream_model",r["id"]),r.get("base_url",""),r.get("api_key_env",""),caps,r.get("fallback_ids",[]),r.get("aliases",[]),r.get("context_window",0),r.get("max_output_tokens",0),r.get("pricing",{}),r.get("extra_headers",{}),"web_search" in caps))
 for x in (_env("TEXT","text","responses"),_env("IMAGE","image","images"),_env("EMBEDDING","embedding","embeddings")):
  if x:specs.insert(0,x)
 out={}
 for s in specs:
  out[s.id.lower()]=s
  for a in s.aliases:out[a.lower()]=s
 if not out:raise RuntimeError("no models configured")
 return out
