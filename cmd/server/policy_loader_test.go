package main

import (
	"context"
	"strings"
	"testing"
)

// The in-memory policy loader is the ENFORCED Model Armor tier (the baked
// agent-guardrail-mapping.yaml is copied into the image but never read, see
// its ENFORCED block). These tests pin the switch so a renamed agent id can
// never silently fall to the balanced default on a kid-facing surface
// (ADR-254 D9: familiar_companion -> companion_chat; the live chat binary
// sends "companion_chat" since the fleet conversion).
func TestDefaultPolicyLoader_ArmorTier(t *testing.T) {
	cfg := runtimeConfig{project: "chora-489812", modelArmorLocation: "asia-southeast1", environment: "dev"}
	loader := newDefaultPolicyLoader(cfg)

	cases := []struct {
		agentID string
		want    string // template id suffix
	}{
		{"companion_chat", "chora-guardrail-strict-dev"},     // the renamed kid-facing chat (ADR-254 D9)
		{"familiar_companion", "chora-guardrail-strict-dev"}, // the pre-rename id, kept through the overlap until G4
		{"qgen_question", "chora-guardrail-strict-dev"},
		{"qgen_critic", "chora-guardrail-strict-dev"},
		{"qgen_renderer", "chora-guardrail-balanced-dev"},         // new id, balanced like every non-listed crew
		{"companion_diagnoser", "chora-guardrail-balanced-dev"},   // predecessor weakness_diagnoser was enforced balanced
		{"kg_explorer", "chora-guardrail-balanced-dev"},
		{"content_recommender", "chora-guardrail-balanced-dev"},
		{"content_moderation", "chora-guardrail-balanced-dev"},
		{"duel_atom_smith", "chora-guardrail-balanced-dev"},
		{"profile_conjurer", "chora-guardrail-balanced-dev"},
		{"never_heard_of_it", "chora-guardrail-balanced-dev"},
	}
	for _, tc := range cases {
		t.Run(tc.agentID, func(t *testing.T) {
			p, err := loader.ResolveAgentPolicy(context.Background(), tc.agentID, tc.agentID, "gemini-2.5-flash", nil)
			if err != nil {
				t.Fatalf("ResolveAgentPolicy(%s): %v", tc.agentID, err)
			}
			if !strings.HasSuffix(p.ArmorTemplate, "/"+tc.want) {
				t.Errorf("agent %q armor template = %q, want suffix %q", tc.agentID, p.ArmorTemplate, tc.want)
			}
		})
	}
}
