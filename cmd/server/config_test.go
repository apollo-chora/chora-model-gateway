// config_test.go: boot-config regressions. First test in this package, added
// for the G1' gap 1 live 404: the embed vendor adapter was wired onto the
// GEMINI Vertex location, and the live deployment sets that to "global",
// where publisher embedding models are not served (predict answers HTTP 404
// HTML). Embeddings need their own regional location config.
package main

import (
	"strings"
	"testing"
)

// setRequiredBootEnv satisfies loadConfig's fail-loud required envs so the
// location assertions can run.
func setRequiredBootEnv(t *testing.T) {
	t.Helper()
	t.Setenv("CHORA_OBSERVABILITY_DSN_SECRET_ID", "test-dsn-secret")
	t.Setenv("CHORA_IDENTITY_GRPC_ADDR", "identity.test:9090")
}

// The embed location must default to the ADR-163 regional endpoint and must
// NOT follow VERTEX_AI_LOCATION: the live gateway legitimately runs Gemini on
// "global", and text-embedding-004 is only served regionally.
func TestLoadConfig_EmbedVertexLocationIndependentOfGeminiLocation(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("VERTEX_AI_LOCATION", "global")
	t.Setenv("CHORA_EMBED_VERTEX_LOCATION", "")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.vertexAILocation != "global" {
		t.Fatalf("vertexAILocation = %q, want global", cfg.vertexAILocation)
	}
	if cfg.embedVertexLocation != "asia-southeast1" {
		t.Fatalf("embedVertexLocation = %q, want the asia-southeast1 default (must not follow VERTEX_AI_LOCATION)", cfg.embedVertexLocation)
	}
}

func TestLoadConfig_EmbedVertexLocationOverride(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_EMBED_VERTEX_LOCATION", "us-central1")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.embedVertexLocation != "us-central1" {
		t.Fatalf("embedVertexLocation = %q, want the env override", cfg.embedVertexLocation)
	}
}

// Fail-closed "budget required" mode must default OFF: an unset flag keeps
// the historical fail-open behaviour (a missing budget window allows).
func TestLoadConfig_BudgetRequiredDefaultOff(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED", "")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.budgetRequired {
		t.Fatalf("budgetRequired = true, want default false")
	}
}

// CHORA_LLM_BUDGET_REQUIRED=true turns a missing active budget window into
// a BLOCK across every provider-egress path.
func TestLoadConfig_BudgetRequiredFlagOn(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED", "true")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if !cfg.budgetRequired {
		t.Fatalf("budgetRequired = false, want true when CHORA_LLM_BUDGET_REQUIRED=true")
	}
}

// CHORA_LLM_BUDGET_REQUIRED_TENANTS parses into a set of canonical tenant
// UUIDs; the global flag stays off so the override's effect is isolated.
func TestLoadConfig_BudgetRequiredTenants_Parsed(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED_TENANTS", "33333333-3333-7333-8333-333333333333, 44444444-4444-7444-8444-444444444444 ")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.budgetRequired {
		t.Fatalf("budgetRequired = true, want the global flag off")
	}
	if len(cfg.budgetRequiredTenants) != 2 {
		t.Fatalf("len(budgetRequiredTenants) = %d, want 2", len(cfg.budgetRequiredTenants))
	}
	if !cfg.budgetRequiredTenants.Contains("33333333-3333-7333-8333-333333333333") {
		t.Errorf("listed tenant 33333333-3333-7333-8333-333333333333 missing from budgetRequiredTenants")
	}
	if !cfg.budgetRequiredTenants.Contains("44444444-4444-7444-8444-444444444444") {
		t.Errorf("listed tenant 44444444-4444-7444-8444-444444444444 missing from budgetRequiredTenants")
	}
}

// Unset ⇒ empty set, no error.
func TestLoadConfig_BudgetRequiredTenants_DefaultEmpty(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED_TENANTS", "")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if len(cfg.budgetRequiredTenants) != 0 {
		t.Fatalf("budgetRequiredTenants = %v, want empty", cfg.budgetRequiredTenants)
	}
}

// A malformed tenant UUID fails the boot — a typo must not silently un-gate
// a tenant (mirrors the CHORA_DEMO_MANA_ALLOWED_GCIDS contract in
// chora-identity).
func TestLoadConfig_BudgetRequiredTenants_Malformed_FailsBoot(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED_TENANTS", "33333333-3333-7333-8333-333333333333,not-a-uuid")

	_, err := loadConfig()
	if err == nil {
		t.Fatal("loadConfig must fail on a malformed tenant UUID in CHORA_LLM_BUDGET_REQUIRED_TENANTS")
	}
	if !strings.Contains(err.Error(), "CHORA_LLM_BUDGET_REQUIRED_TENANTS") {
		t.Errorf("err = %v, want it to name CHORA_LLM_BUDGET_REQUIRED_TENANTS", err)
	}
}

// An empty entry (trailing comma) fails the boot too.
func TestLoadConfig_BudgetRequiredTenants_EmptyEntry_FailsBoot(t *testing.T) {
	setRequiredBootEnv(t)
	t.Setenv("CHORA_LLM_BUDGET_REQUIRED_TENANTS", "33333333-3333-7333-8333-333333333333,")

	_, err := loadConfig()
	if err == nil {
		t.Fatal("loadConfig must fail on an empty entry in CHORA_LLM_BUDGET_REQUIRED_TENANTS")
	}
	if !strings.Contains(err.Error(), "CHORA_LLM_BUDGET_REQUIRED_TENANTS") {
		t.Errorf("err = %v, want it to name CHORA_LLM_BUDGET_REQUIRED_TENANTS", err)
	}
}
