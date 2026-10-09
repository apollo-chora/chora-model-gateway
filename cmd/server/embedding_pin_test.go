package main

import (
	"context"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/apollo-chora/chora-model-gateway/internal/registry"
)

// fakePinEmbedder is a minimal domain.EmbeddingClient whose Family() lets a
// test stand in for the wired adapter (openai = registry-driven, vertex =
// publisher adapter that ignores the registry's upstream_model).
type fakePinEmbedder struct{ family domain.VendorFamily }

func (f fakePinEmbedder) Family() domain.VendorFamily { return f.family }

func (f fakePinEmbedder) EmbedText(context.Context, domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	return domain.EmbedVendorResponse{}, nil
}

func pinRegistry(t *testing.T, spec registry.ModelSpec) registry.Registry {
	t.Helper()
	reg, err := registry.Finalize([]registry.ModelSpec{
		spec,
		{ID: "gpt-4o", Provider: "openai", UpstreamModel: "gpt-4o", BaseURL: "https://api.openai.com/v1"},
	})
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	return reg
}

func approvedPinSpec() registry.ModelSpec {
	return registry.ModelSpec{
		ID:            defaultEmbeddingPinLogicalID,
		Provider:      "openai",
		UpstreamModel: defaultEmbeddingPinUpstreamModel,
		BaseURL:       "https://openrouter.ai/api/v1",
		Capabilities:  []string{"embeddings"},
	}
}

func enabledPin() embeddingPinConfig {
	return embeddingPinConfig{
		Enabled:       true,
		LogicalID:     defaultEmbeddingPinLogicalID,
		UpstreamModel: defaultEmbeddingPinUpstreamModel,
		ProviderHost:  defaultEmbeddingPinProviderHost,
	}
}

func TestVerifyEmbeddingRoutePin_DisabledIsANoOp(t *testing.T) {
	// A registry that maps the logical id to a DIFFERENT upstream is fine while
	// the pin is off — ordinary deployments are unaffected.
	spec := approvedPinSpec()
	spec.UpstreamModel = "some-other-model"
	reg := pinRegistry(t, spec)
	if err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyVertexGemini}}, embeddingPinConfig{}); err != nil {
		t.Fatalf("disabled pin must not fail: %v", err)
	}
}

func TestVerifyEmbeddingRoutePin_ApprovedRoutePasses(t *testing.T) {
	reg := pinRegistry(t, approvedPinSpec())
	if err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin()); err != nil {
		t.Fatalf("approved route must pass: %v", err)
	}
}

func TestVerifyEmbeddingRoutePin_UpstreamMismatchFails(t *testing.T) {
	spec := approvedPinSpec()
	spec.UpstreamModel = "openai/text-embedding-3-large"
	reg := pinRegistry(t, spec)
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin())
	if err == nil {
		t.Fatal("a different upstream model must fail the boot")
	}
}

func TestVerifyEmbeddingRoutePin_FallbackFails(t *testing.T) {
	spec := approvedPinSpec()
	spec.FallbackIDs = []string{"gpt-4o"}
	reg := pinRegistry(t, spec)
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin())
	if err == nil {
		t.Fatal("a declared fallback must fail the boot")
	}
}

func TestVerifyEmbeddingRoutePin_ProviderHostMismatchFails(t *testing.T) {
	spec := approvedPinSpec()
	spec.BaseURL = "https://api.somewhere-else.example/v1"
	reg := pinRegistry(t, spec)
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin())
	if err == nil {
		t.Fatal("a different provider host must fail the boot")
	}
}

func TestVerifyEmbeddingRoutePin_MissingLogicalIDFails(t *testing.T) {
	reg := pinRegistry(t, approvedPinSpec())
	cfg := enabledPin()
	cfg.LogicalID = "not-in-the-registry"
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, cfg)
	if err == nil {
		t.Fatal("an unknown logical id must fail the boot")
	}
}

// The wired adapter must be the one the pinned entry declares. This is the
// check that makes the registry assertion bind the EFFECTIVE route: the Embed
// flow dispatches through the EmbeddingClient port, which never reads the
// registry's upstream_model.
func TestVerifyEmbeddingRoutePin_AdapterMismatchFails(t *testing.T) {
	reg := pinRegistry(t, approvedPinSpec())
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyVertexGemini}}, enabledPin())
	if err == nil {
		t.Fatal("an adapter that cannot honour the registry upstream must fail the boot")
	}
}

func TestLoadEmbeddingPinConfig_DefaultsAndStrictness(t *testing.T) {
	t.Setenv("CHORA_EMBEDDING_PIN_ENABLED", "")
	cfg, err := loadEmbeddingPinConfig()
	if err != nil {
		t.Fatalf("loadEmbeddingPinConfig: %v", err)
	}
	if cfg.Enabled {
		t.Fatal("the pin must default to disabled")
	}
	if cfg.LogicalID != defaultEmbeddingPinLogicalID || cfg.UpstreamModel != defaultEmbeddingPinUpstreamModel {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if cfg.domainPin() != nil {
		t.Fatal("a disabled pin must not produce a domain pin")
	}

	// Enabled: the defaults are applied and a domain pin is produced.
	t.Setenv("CHORA_EMBEDDING_PIN_ENABLED", "true")
	cfg, err = loadEmbeddingPinConfig()
	if err != nil {
		t.Fatalf("loadEmbeddingPinConfig: %v", err)
	}
	if !cfg.Enabled || cfg.domainPin() == nil {
		t.Fatal("an enabled pin must produce a domain pin")
	}

	// Enabled with a blank upstream fails the boot.
	t.Setenv("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", " ")
	if _, err := loadEmbeddingPinConfig(); err == nil {
		t.Fatal("a blank approved model id must fail the boot")
	}
}
