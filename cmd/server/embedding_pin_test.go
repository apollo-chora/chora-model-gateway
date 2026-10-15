package main

import (
	"context"
	"strings"
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

// The pin is env-only (CHORA_EMBEDDING_PIN_*): there is no built-in default
// model, so the tests name their own neutral fixtures.
const (
	testPinLogicalID     = "chora-embedding-test"
	testPinUpstreamModel = "test-embedding-upstream"
	testPinProviderHost  = "openrouter.ai"
)

func approvedPinSpec() registry.ModelSpec {
	return registry.ModelSpec{
		ID:            testPinLogicalID,
		Provider:      "openai",
		UpstreamModel: testPinUpstreamModel,
		BaseURL:       "https://openrouter.ai/api/v1",
		Capabilities:  []string{"embeddings"},
		APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
	}
}

// approvedPinEnv sets the credential the approved route's reference names, so
// the boot check can verify its availability without a value ever appearing in
// a test fixture or a log line.
func approvedPinEnv(t *testing.T) {
	t.Helper()
	t.Setenv("EMBEDDING_LLM_API_KEY", "test-credential")
}

func enabledPin() embeddingPinConfig {
	return embeddingPinConfig{
		Enabled:       true,
		LogicalID:     testPinLogicalID,
		UpstreamModel: testPinUpstreamModel,
		ProviderHost:  testPinProviderHost,
		Dimensions:    defaultEmbeddingPinDimensions,
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
	approvedPinEnv(t)
	reg := pinRegistry(t, approvedPinSpec())
	if err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin()); err != nil {
		t.Fatalf("approved route must pass: %v", err)
	}
}

// The approved endpoint authenticates every request, so a pinned entry that
// names no credential reference — or one that resolves to nothing — fails the
// boot instead of dispatching anonymously. The refusal names the reference,
// never its value.
func TestVerifyEmbeddingRoutePin_RequiresAnAvailableCredential(t *testing.T) {
	reg := pinRegistry(t, approvedPinSpec())

	// No reference at all.
	spec := approvedPinSpec()
	spec.APIKeyEnv = ""
	noRef := pinRegistry(t, spec)
	if err := verifyEmbeddingRoutePin(noRef, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin()); err == nil {
		t.Fatal("an entry with no credential reference must fail the boot")
	}

	// A reference that resolves to nothing.
	t.Setenv("EMBEDDING_LLM_API_KEY", "")
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyOpenAI}}, enabledPin())
	if err == nil {
		t.Fatal("a reference resolving to an empty value must fail the boot")
	}
	if !strings.Contains(err.Error(), "EMBEDDING_LLM_API_KEY") {
		t.Fatalf("the refusal must name the reference (never the value); got %v", err)
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
	approvedPinEnv(t)
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
	approvedPinEnv(t)
	reg := pinRegistry(t, approvedPinSpec())
	err := verifyEmbeddingRoutePin(reg, []domain.EmbeddingClient{fakePinEmbedder{family: domain.VendorFamilyVertexGemini}}, enabledPin())
	if err == nil {
		t.Fatal("an adapter that cannot honour the registry upstream must fail the boot")
	}
}

func TestLoadEmbeddingPinConfig_DefaultsAndStrictness(t *testing.T) {
	t.Setenv("CHORA_EMBEDDING_PIN_ENABLED", "")
	t.Setenv("CHORA_EMBEDDING_PIN_DIMENSIONS", "")
	t.Setenv("CHORA_EMBEDDING_PIN_LOGICAL_ID", "")
	t.Setenv("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", "")
	t.Setenv("CHORA_EMBEDDING_PIN_PROVIDER_HOST", "")
	cfg, err := loadEmbeddingPinConfig()
	if err != nil {
		t.Fatalf("loadEmbeddingPinConfig: %v", err)
	}
	if cfg.Enabled {
		t.Fatal("the pin must default to disabled")
	}
	if cfg.LogicalID != "" || cfg.UpstreamModel != "" || cfg.ProviderHost != "" {
		t.Fatalf("every model/host pin field is env-only and has no default; got %+v", cfg)
	}
	if cfg.domainPin() != nil {
		t.Fatal("a disabled pin must not produce a domain pin")
	}

	// Enabled: the env-provided values are used verbatim and a domain pin is
	// produced. The pin carries the configured vector length, so the Embed flow
	// can require it without a registry entry declaring it.
	t.Setenv("CHORA_EMBEDDING_PIN_ENABLED", "true")
	t.Setenv("CHORA_EMBEDDING_PIN_LOGICAL_ID", testPinLogicalID)
	t.Setenv("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", testPinUpstreamModel)
	t.Setenv("CHORA_EMBEDDING_PIN_PROVIDER_HOST", testPinProviderHost)
	cfg, err = loadEmbeddingPinConfig()
	if err != nil {
		t.Fatalf("loadEmbeddingPinConfig: %v", err)
	}
	if !cfg.Enabled || cfg.domainPin() == nil {
		t.Fatal("an enabled pin must produce a domain pin")
	}
	if cfg.LogicalID != testPinLogicalID || cfg.UpstreamModel != testPinUpstreamModel || cfg.ProviderHost != testPinProviderHost {
		t.Fatalf("the pin must carry the env-provided values; got %+v", cfg)
	}
	if cfg.Dimensions != defaultEmbeddingPinDimensions {
		t.Fatalf("the pin must default to %d dimensions; got %d", defaultEmbeddingPinDimensions, cfg.Dimensions)
	}
	if got := cfg.domainPin().ExpectedDimensions; got != int32(defaultEmbeddingPinDimensions) {
		t.Fatalf("the domain pin must carry the approved dimension; got %d", got)
	}

	// A non-positive dimension fails the boot rather than verifying nothing.
	t.Setenv("CHORA_EMBEDDING_PIN_DIMENSIONS", "0")
	if _, err := loadEmbeddingPinConfig(); err == nil {
		t.Fatal("a non-positive approved dimension must fail the boot")
	}
	t.Setenv("CHORA_EMBEDDING_PIN_DIMENSIONS", "not-a-number")
	if _, err := loadEmbeddingPinConfig(); err == nil {
		t.Fatal("a non-numeric approved dimension must fail the boot")
	}

	// Enabled with a blank upstream, or a blank logical id, fails the boot: the
	// pin is env-only and a half-specified pin would verify nothing.
	t.Setenv("CHORA_EMBEDDING_PIN_DIMENSIONS", "")
	t.Setenv("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", " ")
	if _, err := loadEmbeddingPinConfig(); err == nil {
		t.Fatal("a blank approved model id must fail the boot")
	}
	t.Setenv("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", testPinUpstreamModel)
	t.Setenv("CHORA_EMBEDDING_PIN_LOGICAL_ID", " ")
	if _, err := loadEmbeddingPinConfig(); err == nil {
		t.Fatal("a blank pinned logical id must fail the boot")
	}
}
