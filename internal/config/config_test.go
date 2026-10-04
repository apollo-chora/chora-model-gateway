package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// writeRegistry materialises a registry file in a temp dir and returns its path.
func writeRegistry(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	return path
}

func TestLoadRegistry_ResolvesEntries(t *testing.T) {
	reg, err := LoadRegistry(writeRegistry(t, `
models:
  - id: Fast
    provider: openai
    upstream_model: gpt-4o-mini
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    context_window: 128000
    max_output_tokens: 16384
    capabilities: [chat, tools]
    aliases: [quick, speedy]
    fallback_ids: [big]
    pricing:
      input_per_mtok_usd_micros: 150000
      output_per_mtok_usd_micros: 600000

  - id: big
    provider: openai
    base_url: https://api.openai.com/v1
    capabilities: [chat]
`))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}

	// The lookup is case-insensitive: callers pass "Fast" and "fast" and both
	// must resolve, because model names arrive from JSON bodies.
	for _, name := range []string{"Fast", "fast", "FAST"} {
		got, ok := reg.Lookup(domain.LogicalModelID(name))
		if !ok {
			t.Fatalf("Lookup(%q) missed", name)
		}
		if got.UpstreamModel != "gpt-4o-mini" {
			t.Errorf("Lookup(%q).UpstreamModel = %q, want the alias target", name, got.UpstreamModel)
		}
		if got.APIKeyRef != "OPENAI_API_KEY" {
			t.Errorf("Lookup(%q).APIKeyRef = %q", name, got.APIKeyRef)
		}
	}

	// An id defaults to itself as the upstream model.
	big, ok := reg.Lookup("big")
	if !ok {
		t.Fatal("Lookup(big) missed")
	}
	if big.UpstreamModel != "big" {
		t.Errorf("big.UpstreamModel = %q, want it to default to the id", big.UpstreamModel)
	}
	// An entry with no capabilities list means chat only.
	if !big.Supports(domain.CapabilityChat) || big.Supports(domain.CapabilityImage) {
		t.Errorf("big capabilities = %v, want chat-only by default", big.Capabilities)
	}

	// Aliases answer to the same target.
	if _, ok := reg.Lookup("speedy"); !ok {
		t.Error("alias 'speedy' did not resolve")
	}

	// The declared fallback chain resolves to real targets.
	fbs := reg.FallbacksFor("fast")
	if len(fbs) != 1 || fbs[0].LogicalModelID != "big" {
		t.Errorf("FallbacksFor(fast) = %v, want [big]", fbs)
	}

	// Pricing is attached per model name.
	p, ok := reg.PricingFor("fast")
	if !ok || p.InputPerMTokUSDMicros != 150000 {
		t.Errorf("PricingFor(fast) = %+v", p)
	}
}

func TestLoadRegistry_EndpointResolution(t *testing.T) {
	reg, err := LoadRegistry(writeRegistry(t, `
models:
  - id: joined
    provider: openai
    base_url: https://api.openai.com/v1
    capabilities: [chat, image, embeddings]

  - id: trailing
    provider: openai
    base_url: https://api.openai.com/v1/
    capabilities: [chat]

  - id: relative-path
    provider: openai
    base_url: https://api.openai.com/v1
    chat_completions_path: generate
    capabilities: [chat]

  - id: absolute-override
    provider: openai
    base_url: https://ignored.example.com/v1
    chat_completions_path: https://llm.internal:8443/generate
    capabilities: [chat]

  - id: claude
    provider: anthropic
    base_url: https://api.anthropic.com
    capabilities: [chat]
`))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}

	cases := []struct {
		model          string
		wantChat       string
		wantImages     string
		wantEmbeddings string
		wantPrimary    string
	}{
		{
			model:          "joined",
			wantChat:       "https://api.openai.com/v1/chat/completions",
			wantImages:     "https://api.openai.com/v1/images/generations",
			wantEmbeddings: "https://api.openai.com/v1/embeddings",
			wantPrimary:    "https://api.openai.com/v1/chat/completions",
		},
		{
			// A trailing slash on base_url must not produce a doubled slash.
			model:          "trailing",
			wantChat:       "https://api.openai.com/v1/chat/completions",
			wantImages:     "https://api.openai.com/v1/images/generations",
			wantEmbeddings: "https://api.openai.com/v1/embeddings",
			wantPrimary:    "https://api.openai.com/v1/chat/completions",
		},
		{
			// A relative path without a leading slash still joins cleanly.
			model:          "relative-path",
			wantChat:       "https://api.openai.com/v1/generate",
			wantImages:     "https://api.openai.com/v1/images/generations",
			wantEmbeddings: "https://api.openai.com/v1/embeddings",
			wantPrimary:    "https://api.openai.com/v1/generate",
		},
		{
			// An absolute path wins outright and base_url is ignored: this is
			// the "fully custom chat-completions endpoint" case.
			model:          "absolute-override",
			wantChat:       "https://llm.internal:8443/generate",
			wantImages:     "https://ignored.example.com/v1/images/generations",
			wantEmbeddings: "https://ignored.example.com/v1/embeddings",
			wantPrimary:    "https://llm.internal:8443/generate",
		},
		{
			// An anthropic entry's primary endpoint is /v1/messages, NOT
			// OpenAI's /chat/completions. Getting this wrong would send
			// Anthropic traffic to a path it does not serve.
			model:          "claude",
			wantChat:       "https://api.anthropic.com/chat/completions",
			wantImages:     "https://api.anthropic.com/images/generations",
			wantEmbeddings: "https://api.anthropic.com/embeddings",
			wantPrimary:    "https://api.anthropic.com/v1/messages",
		},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			target, ok := reg.Lookup(domain.LogicalModelID(tc.model))
			if !ok {
				t.Fatalf("Lookup(%q) missed", tc.model)
			}
			if got := target.ChatCompletionsURL(); got != tc.wantChat {
				t.Errorf("ChatCompletionsURL = %q, want %q", got, tc.wantChat)
			}
			if got := target.ImagesURL(); got != tc.wantImages {
				t.Errorf("ImagesURL = %q, want %q", got, tc.wantImages)
			}
			if got := target.EmbeddingsURL(); got != tc.wantEmbeddings {
				t.Errorf("EmbeddingsURL = %q, want %q", got, tc.wantEmbeddings)
			}
			if got := target.PrimaryDispatchURL(); got != tc.wantPrimary {
				t.Errorf("PrimaryDispatchURL = %q, want %q", got, tc.wantPrimary)
			}
		})
	}
}

func TestLoadRegistry_RejectsBadConfig(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "no models",
			body:    "models: []\n",
			wantErr: "declares no models",
		},
		{
			name:    "missing id",
			body:    "models:\n  - provider: openai\n    base_url: https://x.test/v1\n",
			wantErr: "id required",
		},
		{
			name:    "missing base_url",
			body:    "models:\n  - id: m\n    provider: openai\n",
			wantErr: "base_url required",
		},
		{
			name:    "unknown provider",
			body:    "models:\n  - id: m\n    provider: vertex\n    base_url: https://x.test/v1\n",
			wantErr: "is not one of: openai, anthropic",
		},
		{
			name:    "unknown capability",
			body:    "models:\n  - id: m\n    provider: openai\n    base_url: https://x.test/v1\n    capabilities: [chat, telepathy]\n",
			wantErr: `capability "telepathy"`,
		},
		{
			name:    "output ceiling exceeds context",
			body:    "models:\n  - id: m\n    provider: openai\n    base_url: https://x.test/v1\n    context_window: 100\n    max_output_tokens: 200\n",
			wantErr: "exceeds context_window",
		},
		{
			name:    "unknown fallback",
			body:    "models:\n  - id: m\n    provider: openai\n    base_url: https://x.test/v1\n    fallback_ids: [ghost]\n",
			wantErr: "not in the registry",
		},
		{
			name:    "scheme-relative path",
			body:    "models:\n  - id: m\n    provider: openai\n    base_url: https://x.test/v1\n    chat_completions_path: \"//llm.internal/generate\"\n",
			wantErr: "missing its scheme",
		},
		{
			name:    "alias collision with a different upstream",
			body:    "models:\n  - id: a\n    provider: openai\n    base_url: https://x.test/v1\n    upstream_model: one\n  - id: b\n    provider: openai\n    base_url: https://x.test/v1\n    upstream_model: two\n    aliases: [a]\n",
			wantErr: "declared twice",
		},
		{
			name:    "malformed yaml",
			body:    "models:\n  - id: [unclosed\n",
			wantErr: "parse registry",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadRegistry(writeRegistry(t, tc.body))
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadRegistry_MissingFile(t *testing.T) {
	if _, err := LoadRegistry(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Fatal("expected an error for a missing registry file")
	}
}

func TestRegistry_Known(t *testing.T) {
	reg, err := LoadRegistry(writeRegistry(t, `
models:
  - id: only
    provider: openai
    base_url: https://x.test/v1
`))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if !reg.Known("only") {
		t.Error("Known(only) = false")
	}
	if reg.Known("missing") {
		t.Error("Known(missing) = true")
	}
	if got := len(reg.Names()); got != 1 {
		t.Errorf("Names() = %v, want one entry", reg.Names())
	}
}

func TestLoadConfig_RequiresDatabaseAndTenant(t *testing.T) {
	// Both are load-bearing: the budget row is tenant-scoped, so a gateway
	// without them cannot settle anything. Refusing at boot beats refusing
	// every request.
	for _, tc := range []struct{ name, db, tenant string }{
		{"no db", "", "t"},
		{"no tenant", "postgres://x", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("CHORA_DATABASE_URL", tc.db)
			t.Setenv("CHORA_DEFAULT_TENANT_ID", tc.tenant)
			_, err := LoadConfig()
			if err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestLoadConfig_DefaultsAndOverrides(t *testing.T) {
	t.Setenv("CHORA_DATABASE_URL", "postgres://localhost/db")
	t.Setenv("CHORA_DEFAULT_TENANT_ID", "tenant-1")
	t.Setenv("CHORA_VENDOR_HTTP_TIMEOUT", "45s")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.VendorHTTPTimeout.Seconds() != 45 {
		t.Errorf("vendor timeout = %v, want 45s", cfg.VendorHTTPTimeout)
	}
	if cfg.BootstrapTimeout == 0 {
		t.Error("bootstrap timeout defaulted to zero")
	}
	// GCID falls back to the tenant rather than being empty, so a ledger row
	// always has an actor.
	if cfg.DefaultGCID != "tenant-1" {
		t.Errorf("default gcid = %q, want it to fall back to the tenant", cfg.DefaultGCID)
	}
}

func TestLoadConfig_RejectsBadDuration(t *testing.T) {
	t.Setenv("CHORA_DATABASE_URL", "postgres://localhost/db")
	t.Setenv("CHORA_DEFAULT_TENANT_ID", "t")
	t.Setenv("CHORA_VENDOR_HTTP_TIMEOUT", "not-a-duration")
	if _, err := LoadConfig(); err == nil {
		t.Fatal("expected an error for an unparseable duration")
	}
}

// TestShippedRegistryLoads guards the file docker-compose actually mounts.
// A typo in config/models.yaml must fail here rather than on a container
// restart an operator has to diagnose from a log.
func TestShippedRegistryLoads(t *testing.T) {
	if _, err := LoadRegistry(filepath.Join("..", "..", "config", "models.yaml")); err != nil {
		t.Fatalf("the shipped config/models.yaml does not load: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Grounding blocks
// ---------------------------------------------------------------------------

func TestLoadRegistry_Grounding(t *testing.T) {
	reg, err := LoadRegistry(writeRegistry(t, `
models:
  - id: defaulted
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [chat, web_search]
    grounding: {}

  - id: fully-spec
    provider: anthropic
    base_url: https://api.anthropic.com
    capabilities: [chat, web_search]
    grounding:
      surface: messages
      tool_type: web_search_20260209
      tool_name: search
      max_uses: 3
      extra_tool_fields:
        allowed_domains: ["example.com"]

  - id: chat-surface
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [web_search]
    grounding:
      surface: chat_completions
      responses_path: /gen/responses
`))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}

	// An empty block still resolves by provider family.
	d, ok := reg.Lookup("defaulted")
	if !ok {
		t.Fatal("defaulted did not resolve")
	}
	if d.Grounding == nil {
		t.Fatal("grounding block was dropped")
	}
	if got := d.Grounding.EffectiveSurface(d.Vendor); got != domain.SurfaceResponses {
		t.Errorf("surface = %q, want responses", got)
	}
	if got := d.Grounding.EffectiveToolType(domain.SurfaceResponses); got != "web_search" {
		t.Errorf("tool type = %q, want web_search", got)
	}
	if got := d.GroundingURL(); got != "https://api.example.test/v1/responses" {
		t.Errorf("grounding URL = %q", got)
	}

	f, _ := reg.Lookup("fully-spec")
	if got := f.Grounding.EffectiveToolType(domain.SurfaceMessages); got != "web_search_20260209" {
		t.Errorf("explicit tool type = %q", got)
	}
	if got := f.Grounding.EffectiveToolName(); got != "search" {
		t.Errorf("tool name = %q", got)
	}
	if f.Grounding.MaxUses != 3 {
		t.Errorf("max uses = %d", f.Grounding.MaxUses)
	}
	if _, present := f.Grounding.ExtraToolFields["allowed_domains"]; !present {
		t.Errorf("extra tool fields dropped: %+v", f.Grounding.ExtraToolFields)
	}
	// An anthropic entry grounds on the messages endpoint.
	if got := f.GroundingURL(); got != "https://api.anthropic.com/v1/messages" {
		t.Errorf("anthropic grounding URL = %q", got)
	}

	c, _ := reg.Lookup("chat-surface")
	if got := c.GroundingURL(); got != "https://api.example.test/v1/chat/completions" {
		t.Errorf("chat surface URL = %q", got)
	}
}

func TestLoadRegistry_GroundingRejectsBadConfig(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			// Declaring the tool without the capability would accept the file
			// and then refuse every grounded call at dispatch.
			name: "grounding block without the capability",
			body: `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat]
    grounding: {}`,
			wantErr: "capabilities` does not include",
		},
		{
			name: "unknown surface",
			body: `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, web_search]
    grounding:
      surface: telepathy`,
			wantErr: "is not one of: responses, chat_completions, messages",
		},
		{
			name: "negative max_uses",
			body: `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, web_search]
    grounding:
      max_uses: -1`,
			wantErr: "max_uses cannot be negative",
		},
		{
			name: "unknown capability spelling",
			body: `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, websearch]
    grounding: {}`,
			wantErr: `capability "websearch"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadRegistry(writeRegistry(t, tc.body))
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadRegistry_NoGroundingBlockMeansNoGrounding(t *testing.T) {
	// The default must be "cannot ground", so a caller asking for search gets
	// a refusal instead of an ungrounded answer that looks researched.
	reg, err := LoadRegistry(writeRegistry(t, `
models:
  - id: plain
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat]
`))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	target, _ := reg.Lookup("plain")
	if target.Grounding != nil {
		t.Errorf("grounding = %+v, want nil", target.Grounding)
	}
	if got := target.GroundingURL(); got != "" {
		t.Errorf("grounding URL = %q, want empty", got)
	}
	if target.Supports(domain.CapabilityWebSearch) {
		t.Error("a plain entry advertises web_search")
	}
}
