from types import SimpleNamespace
import pytest
from app.compat import embedding_inputs, flatten_responses_input, generation_params, model_entry, wants_grounding

def spec(**kw):
    d={"id":"logical","format":"chat_completions","model":"upstream","base_url":"https://example.test/v1","capabilities":["chat","tools"],"context_window":8192,"max_output_tokens":2048}
    d.update(kw);return SimpleNamespace(**d)

def test_model_entry_preserves_old_metadata():
    assert model_entry(spec())=={"id":"logical","object":"model","created":0,"owned_by":"openai","context_window":8192,"max_output_tokens":2048,"capabilities":["chat","tools"],"base_url":"https://example.test/v1"}

def test_generation_params_preserve_old_chat_fields():
    assert generation_params({"temperature":.2,"top_p":.9,"max_completion_tokens":99,"stop":["x"],"n":2,"seed":7})=={"temperature":.2,"top_p":.9,"max_tokens":99,"stop":["x"],"n":2,"seed":7}

@pytest.mark.parametrize("tool",[{"type":"web_search_preview"},{"name":"web_fetch"}])
def test_grounding_detection(tool):assert wants_grounding([tool])

def test_responses_input_validation():
    assert flatten_responses_input("hello")==("hello",None)
    assert flatten_responses_input([{"role":"user","content":"hello"}])==("hello",None)
    assert flatten_responses_input([{"role":"user","content":"a"},{"role":"assistant","content":"b"}])==("",[{"role":"user","content":"a"},{"role":"assistant","content":"b"}])
    for bad in ("",[]): 
        with pytest.raises(ValueError):flatten_responses_input(bad)

def test_embedding_input_validation():
    assert embedding_inputs("x")==["x"];assert embedding_inputs(["x","y"])==["x","y"]
    for bad in ("",[],["x",""],[1,2]):
        with pytest.raises(ValueError):embedding_inputs(bad)
