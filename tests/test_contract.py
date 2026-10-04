from pathlib import Path
def test_proto_contract_is_v1_compatible():
    p=Path("proto/model_gateway_service.proto").read_text()
    assert "string invocation_id=1" in p
    assert "string dispatch_idempotency_key=18" in p
    assert "rpc Invoke" in p and "rpc Embed" in p
