// Package clients holds chora-model-gateway's outbound gRPC clients.
//
// ManaClient is the WS-1 umbrella-metering port to chora-identity's ManaService
// (chora-contracts/proto/services/identity/v1/mana_service.proto). It satisfies
// middleware.ManaPort:
//   - Quote  → DeductMana(dry_run=true, units=0): resolve catalogue cost + check
//     affordability WITHOUT debiting (the pre-flight gate).
//   - Debit  → DeductMana(dry_run=false, units=0): charge the catalogue cost,
//     idempotent on the invocation id.
//
// An unpriced action_code surfaces from identity as codes.InvalidArgument; the
// client maps that to a soft UnknownAction (un-metered), distinct from a real
// infra error which propagates. Mana costs live in the identity catalogue — the
// gateway carries none (feedback_no_inline_config).
package clients

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	identityv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/identity/v1"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/middleware"
)

// ManaServiceGRPCClient is the minimal slice of identityv1.ManaServiceClient the
// metering port calls. The generated client satisfies it; tests inject a fake.
type ManaServiceGRPCClient interface {
	DeductMana(ctx context.Context, in *identityv1.DeductManaRequest, opts ...grpc.CallOption) (*identityv1.DeductManaResponse, error)
}

// ManaClient adapts ManaServiceGRPCClient to middleware.ManaPort.
type ManaClient struct {
	grpc ManaServiceGRPCClient
}

// NewManaClient wraps a gRPC client (production: identityv1.NewManaServiceClient;
// tests: a fake).
func NewManaClient(g ManaServiceGRPCClient) *ManaClient { return &ManaClient{grpc: g} }

// Compile-time check.
var _ middleware.ManaPort = (*ManaClient)(nil)

// NewManaClientFromAddr dials addr (mesh plaintext — mTLS terminates at the
// Istio sidecar) and returns a ManaClient + its conn-close func. addr MUST be
// non-empty (CHORA_IDENTITY_GRPC_ADDR) — fail-loud per feedback_no_inline_config.
func NewManaClientFromAddr(addr string) (*ManaClient, func() error, error) {
	if addr == "" {
		return nil, nil, errors.New("clients.ManaClient: CHORA_IDENTITY_GRPC_ADDR required")
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, fmt.Errorf("clients.ManaClient: dial %q: %w", addr, err)
	}
	return NewManaClient(identityv1.NewManaServiceClient(conn)), conn.Close, nil
}

// Quote runs a dry-run affordability check (no debit).
func (c *ManaClient) Quote(ctx context.Context, gcid, tenantID, actionCode string) (middleware.ManaQuote, error) {
	resp, err := c.grpc.DeductMana(ctx, &identityv1.DeductManaRequest{
		Gcid:           gcid,
		ActionCode:     actionCode,
		Units:          0, // catalogue-priced
		IdempotencyKey: dryRunKey(gcid, actionCode),
		TenantId:       tenantID,
		DryRun:         true,
	})
	if err != nil {
		if status.Code(err) == codes.InvalidArgument {
			return middleware.ManaQuote{UnknownAction: true}, nil
		}
		return middleware.ManaQuote{}, fmt.Errorf("mana Quote (dry-run DeductMana): %w", err)
	}
	if !resp.Success {
		return middleware.ManaQuote{
			Affordable:     false,
			RequiredUnits:  resp.RequiredUnits,
			AvailableUnits: resp.CurrentBalanceUnits,
		}, nil
	}
	return middleware.ManaQuote{Affordable: true, AvailableUnits: resp.BalanceAfterUnits}, nil
}

// Debit charges the catalogue cost for action_code, idempotent on idemKey.
func (c *ManaClient) Debit(ctx context.Context, gcid, tenantID, actionCode, idemKey string) (middleware.ManaDebit, error) {
	resp, err := c.grpc.DeductMana(ctx, &identityv1.DeductManaRequest{
		Gcid:           gcid,
		ActionCode:     actionCode,
		Units:          0, // catalogue-priced
		IdempotencyKey: idemKey,
		TenantId:       tenantID,
		DryRun:         false,
	})
	if err != nil {
		if status.Code(err) == codes.InvalidArgument {
			return middleware.ManaDebit{UnknownAction: true}, nil
		}
		return middleware.ManaDebit{}, fmt.Errorf("mana Debit (DeductMana): %w", err)
	}
	if !resp.Success {
		return middleware.ManaDebit{
			Success:           false,
			RequiredUnits:     resp.RequiredUnits,
			BalanceAfterUnits: resp.CurrentBalanceUnits,
		}, nil
	}
	return middleware.ManaDebit{Success: true, BalanceAfterUnits: resp.BalanceAfterUnits}, nil
}

// dryRunKey builds a non-empty idempotency key for the dry-run check. The
// identity domain requires one even though a dry-run writes nothing (the key is
// validated before the dry-run short-circuit, never persisted).
func dryRunKey(gcid, actionCode string) string {
	return fmt.Sprintf("dryrun:%s:%s", gcid, actionCode)
}
