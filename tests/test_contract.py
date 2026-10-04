from pathlib import Path


def test_proto_contract_is_v1_compatible() -> None:
    proto = Path("proto/model_gateway_service.proto").read_text()
    assert "string invocation_id=1" in proto
    assert "string dispatch_idempotency_key=18" in proto
    assert "rpc Invoke" in proto and "rpc Embed" in proto
