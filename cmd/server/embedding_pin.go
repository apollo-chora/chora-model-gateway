package main

import (
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/apollo-chora/chora-model-gateway/internal/registry"
)

// ============================================================================
// Embedding-route pin — demo-deployment verification of the EFFECTIVE loaded
// registry (ChatGPT round 8, conditionally-accepted embedding-cost argument).
//
// The demo pins the embedding LOGICAL id to one approved upstream model and
// asserts at boot that the registry the gateway actually loaded still maps it
// there. The assertion is an exact-id match against the approved upstream, not
// a name-shape rule: a model name is not an authoritative billing policy, and a
// generic "every embedding model must end in :free" guard would be brittle.
//
// Disabled by default (CHORA_EMBEDDING_PIN_ENABLED=false), so ordinary
// deployments are unaffected. When enabled, ANY mismatch fails the boot rather
// than serving a route the operator did not approve.
// ============================================================================

// Defaults for the approved demo route. The upstream model id is the current
// committed value (chora-stack/config/model-gateway/models.prod.yaml); the
// provider host is the OpenRouter endpoint the deployment points
// EMBEDDING_LLM_BASE_URL at.
const (
	defaultEmbeddingPinLogicalID     = "text-embedding-004"
	defaultEmbeddingPinUpstreamModel = "liquid/lfm-2.5-embedding-350m:free"
	defaultEmbeddingPinProviderHost  = "openrouter.ai"
)

// embeddingPinConfig is the parsed CHORA_EMBEDDING_PIN_* configuration.
type embeddingPinConfig struct {
	Enabled       bool
	LogicalID     string
	UpstreamModel string
	ProviderHost  string
}

// loadEmbeddingPinConfig parses the pin configuration. A disabled pin is
// returned as-is (every field may be empty). An ENABLED pin is strict: a blank
// field fails the boot, because a half-specified pin would silently verify
// nothing.
func loadEmbeddingPinConfig() (embeddingPinConfig, error) {
	cfg := embeddingPinConfig{
		Enabled:       os.Getenv("CHORA_EMBEDDING_PIN_ENABLED") == "true",
		LogicalID:     envOr("CHORA_EMBEDDING_PIN_LOGICAL_ID", defaultEmbeddingPinLogicalID),
		UpstreamModel: envOr("CHORA_EMBEDDING_PIN_UPSTREAM_MODEL", defaultEmbeddingPinUpstreamModel),
		ProviderHost:  envOr("CHORA_EMBEDDING_PIN_PROVIDER_HOST", defaultEmbeddingPinProviderHost),
	}
	if !cfg.Enabled {
		return cfg, nil
	}
	for k, v := range map[string]string{
		"CHORA_EMBEDDING_PIN_LOGICAL_ID":     cfg.LogicalID,
		"CHORA_EMBEDDING_PIN_UPSTREAM_MODEL": cfg.UpstreamModel,
	} {
		if strings.TrimSpace(v) == "" {
			return cfg, fmt.Errorf("embedding route pin enabled but %s is empty", k)
		}
	}
	return cfg, nil
}

// domainPin returns the domain-side pin the Embed flow enforces at request
// time (a runtime override that is not the pinned logical id is refused).
// nil when the pin is disabled.
func (c embeddingPinConfig) domainPin() *domain.EmbeddingRoutePin {
	if !c.Enabled {
		return nil
	}
	return &domain.EmbeddingRoutePin{LogicalID: domain.LogicalModelID(c.LogicalID)}
}

// embedderFamilies lists the vendor families of the wired embedding adapters,
// for the boot log.
func embedderFamilies(embedders []domain.EmbeddingClient) []string {
	out := make([]string, 0, len(embedders))
	for _, e := range embedders {
		out = append(out, string(e.Family()))
	}
	return out
}

// verifyEmbeddingRoutePin asserts that the registry the gateway actually
// loaded maps the pinned logical id to the approved upstream model, with no
// fallbacks, and that the wired embedding adapter can honour that entry.
//
// Every failure is returned (never logged-and-continued): the caller fails the
// boot. The single check ChatGPT asked for — "does the effective registry still
// resolve the logical id to the approved model" — is the first assertion; the
// remaining ones close the three other ways the route could drift: a registry
// fallback to a paid model, a different provider endpoint, and an adapter that
// ignores the registry's upstream_model entirely.
func verifyEmbeddingRoutePin(reg registry.Registry, embedders []domain.EmbeddingClient, cfg embeddingPinConfig) error {
	if !cfg.Enabled {
		return nil
	}
	if reg == nil {
		return fmt.Errorf("embedding route pin: no registry loaded")
	}
	spec, ok := reg.Get(cfg.LogicalID)
	if !ok {
		return fmt.Errorf("embedding route pin: logical id %q is not in the effective registry", cfg.LogicalID)
	}
	if spec.UpstreamModel != cfg.UpstreamModel {
		return fmt.Errorf("embedding route pin: logical id %q resolves to upstream %q, expected %q",
			cfg.LogicalID, spec.UpstreamModel, cfg.UpstreamModel)
	}
	if len(spec.FallbackIDs) > 0 {
		return fmt.Errorf("embedding route pin: logical id %q declares fallbacks %v; the pinned route must have none (a fallback can bill a paid model)",
			cfg.LogicalID, spec.FallbackIDs)
	}
	// Provider metadata, where available: the registry entry's base_url when it
	// declares one, else the deployment's EMBEDDING_LLM_BASE_URL (the endpoint
	// the env-role spec would use). Neither present => skip, not fail: the
	// host check is a strengthening, not the assertion.
	host := ""
	if spec.BaseURL != "" {
		host = hostOfURL(spec.BaseURL)
	} else if v := strings.TrimSpace(os.Getenv("EMBEDDING_LLM_BASE_URL")); v != "" {
		host = hostOfURL(v)
	}
	if cfg.ProviderHost != "" && host != "" && host != cfg.ProviderHost {
		return fmt.Errorf("embedding route pin: provider host %q does not match the approved %q", host, cfg.ProviderHost)
	}
	// The wired adapter must be the one the pinned entry declares. Without this
	// the registry assertion is decorative: the Embed flow dispatches through
	// the EmbeddingClient port, which never consults the registry's
	// upstream_model, so a mismatched adapter would serve a DIFFERENT upstream
	// under the approved logical id.
	if len(embedders) == 0 {
		return fmt.Errorf("embedding route pin: no embedding adapter is wired")
	}
	adapters := make(map[domain.VendorFamily]domain.EmbeddingClient, len(embedders))
	for _, e := range embedders {
		adapters[e.Family()] = e
	}
	if expected, ok := providerVendorFamily(spec.Provider); ok {
		if _, wired := adapters[expected]; !wired {
			return fmt.Errorf("embedding route pin: logical id %q declares provider %q (upstream %q) but no embedding adapter for that family is wired, so the registry route would not be honoured",
				cfg.LogicalID, spec.Provider, spec.UpstreamModel)
		}
	}
	return nil
}

// providerVendorFamily maps a registry provider token (openai | anthropic) to
// the domain VendorFamily of the adapter that implements it.
func providerVendorFamily(provider string) (domain.VendorFamily, bool) {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "openai":
		return domain.VendorFamilyOpenAI, true
	case "anthropic":
		return domain.VendorFamilyAnthropic, true
	default:
		return "", false
	}
}

// hostOfURL extracts the host of an absolute http(s) URL, lowercased. Returns
// "" for anything that does not parse.
func hostOfURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return ""
	}
	return strings.ToLower(u.Hostname())
}
