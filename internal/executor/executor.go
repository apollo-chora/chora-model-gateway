// Package executor is the gateway's core execution engine. It composes the
// governance pipeline (companion suspension → mana metering → domain service)
// into a single Execute() path that both the gRPC and HTTP adapters call.
//
// The Executor is the ONLY entry point to the governance pipeline. Adapters
// (gRPC, HTTP) are thin translation layers: they translate their wire format
// (proto, OpenAI) into ExecuteRequest, call Execute(), and translate
// ExecuteResponse back. No governance logic lives in the adapters.
package executor

import (
	"context"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/middleware"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// ExecuteRequest is the transport-agnostic input to Executor.Execute.
// Both the gRPC adapter (proto) and the HTTP adapter (OpenAI format)
// translate their wire format into this type.
type ExecuteRequest = domain.InvokeRequest

// ExecuteResponse is the transport-agnostic output of Executor.Execute.
type ExecuteResponse = domain.InvokeResponse

// Executor is the gateway's core execution engine. It composes the
// governance pipeline (companion suspension → mana metering → domain service)
// into a single Execute() path.
type Executor struct {
	invoker middleware.Next
}

// NewExecutor constructs the Executor. It composes the governance pipeline:
// CompanionSuspension(ManaMetering(domain.Service)).
//
// svc is the domain service (Armor, budget, outbox, fallback).
// mana is the mana metering port (WS-1 umbrella metering).
// manaCfg configures the mana metering decorator.
// gate is the suspension gate (ADR-252 companion containment).
// susCfg configures the suspension decorator.
func NewExecutor(
	svc *domain.Service,
	mana middleware.ManaPort,
	manaCfg middleware.Config,
	gate middleware.SuspensionGate,
	susCfg middleware.SuspensionConfig,
) *Executor {
	metered := middleware.NewManaMetering(svc, mana, manaCfg)
	contained := middleware.NewCompanionSuspension(metered, gate, susCfg)
	return &Executor{invoker: contained}
}

// Execute runs the governance pipeline for one invocation. Both the gRPC
// and HTTP adapters call this method — it is the single execution path.
func (e *Executor) Execute(ctx context.Context, req ExecuteRequest) (ExecuteResponse, error) {
	return e.invoker.Invoke(ctx, req)
}
