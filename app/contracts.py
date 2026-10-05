"""Canonical chora-contracts bindings for the gateway's outbox payload.

The gateway does not own the TokenUsageRecorded event shape — chora-contracts
does. These classes come from the generated ``chora_contracts_gen`` package
(``chora-contracts``, produced by ``buf generate`` from
``proto/events/observability/token_usage.proto``). Do not hand-roll a second
event proto here.
"""

from __future__ import annotations

from chora_contracts_gen.chora.common.v1 import envelope_pb2
from chora_contracts_gen.events.observability import token_usage_pb2

EventEnvelope = envelope_pb2.EventEnvelope
TokenUsageRecorded = token_usage_pb2.TokenUsageRecorded

__all__ = ["EventEnvelope", "TokenUsageRecorded"]
