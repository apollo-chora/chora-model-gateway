package clients_test

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	identityv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/identity/v1"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/clients"
)

type fakeGRPC struct {
	lastReq *identityv1.DeductManaRequest
	resp    *identityv1.DeductManaResponse
	err     error
}

func (f *fakeGRPC) DeductMana(_ context.Context, in *identityv1.DeductManaRequest, _ ...grpc.CallOption) (*identityv1.DeductManaResponse, error) {
	f.lastReq = in
	return f.resp, f.err
}

func TestManaClient_Quote_Affordable(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: true, BalanceAfterUnits: 95}}
	c := clients.NewManaClient(g)

	q, err := c.Quote(context.Background(), "gcid-1", "tenant-1", "familiar_chat_turn_basic")
	if err != nil {
		t.Fatalf("Quote: %v", err)
	}
	if !q.Affordable {
		t.Errorf("Affordable = false, want true")
	}
	if q.AvailableUnits != 95 {
		t.Errorf("AvailableUnits = %d, want 95", q.AvailableUnits)
	}
	if !g.lastReq.DryRun {
		t.Errorf("DeductMana DryRun = false, want true (Quote is a dry-run)")
	}
	if g.lastReq.Units != 0 {
		t.Errorf("Units = %d, want 0 (catalogue-priced)", g.lastReq.Units)
	}
	if g.lastReq.ActionCode != "familiar_chat_turn_basic" || g.lastReq.Gcid != "gcid-1" || g.lastReq.TenantId != "tenant-1" {
		t.Errorf("req fields not threaded: %+v", g.lastReq)
	}
	if g.lastReq.IdempotencyKey == "" {
		t.Errorf("IdempotencyKey empty — the identity domain requires a non-empty key even for dry-run")
	}
}

func TestManaClient_Quote_Insufficient(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: false, RequiredUnits: 30, CurrentBalanceUnits: 5}}
	c := clients.NewManaClient(g)

	q, err := c.Quote(context.Background(), "g", "t", "familiar_chat_turn_premium")
	if err != nil {
		t.Fatalf("Quote: %v", err)
	}
	if q.Affordable {
		t.Errorf("Affordable = true, want false")
	}
	if q.RequiredUnits != 30 || q.AvailableUnits != 5 {
		t.Errorf("required/available = %d/%d, want 30/5", q.RequiredUnits, q.AvailableUnits)
	}
}

func TestManaClient_Quote_UnknownActionInvalidArgument(t *testing.T) {
	g := &fakeGRPC{err: status.Error(codes.InvalidArgument, "user_mana: unknown action_code")}
	c := clients.NewManaClient(g)

	q, err := c.Quote(context.Background(), "g", "t", "nope")
	if err != nil {
		t.Fatalf("Quote: %v (unknown action must be a soft UnknownAction, not an error)", err)
	}
	if !q.UnknownAction {
		t.Errorf("UnknownAction = false, want true on InvalidArgument")
	}
}

func TestManaClient_Quote_OtherErrorPropagates(t *testing.T) {
	g := &fakeGRPC{err: status.Error(codes.Unavailable, "identity down")}
	c := clients.NewManaClient(g)

	if _, err := c.Quote(context.Background(), "g", "t", "familiar_chat_turn_basic"); err == nil {
		t.Fatalf("expected Unavailable to propagate as an error")
	}
}

func TestManaClient_Debit_Success(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: true, BalanceAfterUnits: 70}}
	c := clients.NewManaClient(g)

	d, err := c.Debit(context.Background(), "g", "t", "familiar_chat_turn_basic", "inv-7")
	if err != nil {
		t.Fatalf("Debit: %v", err)
	}
	if !d.Success {
		t.Errorf("Success = false, want true")
	}
	if d.BalanceAfterUnits != 70 {
		t.Errorf("BalanceAfterUnits = %d, want 70", d.BalanceAfterUnits)
	}
	if g.lastReq.DryRun {
		t.Errorf("Debit DryRun = true, want false (real debit)")
	}
	if g.lastReq.IdempotencyKey != "inv-7" {
		t.Errorf("IdempotencyKey = %q, want inv-7 (the invocation id)", g.lastReq.IdempotencyKey)
	}
	if g.lastReq.Units != 0 {
		t.Errorf("Units = %d, want 0", g.lastReq.Units)
	}
}

func TestManaClient_Debit_Insufficient(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: false, RequiredUnits: 200, CurrentBalanceUnits: 10}}
	c := clients.NewManaClient(g)

	d, err := c.Debit(context.Background(), "g", "t", "boss_challenge_atom_gen", "inv-9")
	if err != nil {
		t.Fatalf("Debit: %v", err)
	}
	if d.Success {
		t.Errorf("Success = true, want false")
	}
	if d.RequiredUnits != 200 || d.BalanceAfterUnits != 10 {
		t.Errorf("required/balance = %d/%d, want 200/10", d.RequiredUnits, d.BalanceAfterUnits)
	}
}

func TestManaClient_Debit_UnknownActionInvalidArgument(t *testing.T) {
	g := &fakeGRPC{err: status.Error(codes.InvalidArgument, "unknown action_code")}
	c := clients.NewManaClient(g)

	d, err := c.Debit(context.Background(), "g", "t", "nope", "inv-1")
	if err != nil {
		t.Fatalf("Debit: %v (unknown action must be soft)", err)
	}
	if !d.UnknownAction {
		t.Errorf("UnknownAction = false, want true")
	}
}

func TestManaClient_Debit_OtherErrorPropagates(t *testing.T) {
	g := &fakeGRPC{err: errors.New("boom")}
	c := clients.NewManaClient(g)
	if _, err := c.Debit(context.Background(), "g", "t", "x", "i"); err == nil {
		t.Fatalf("expected error to propagate")
	}
}
