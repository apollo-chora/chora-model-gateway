package domain

import (
	"encoding/json"
	"sort"
	"strings"
)

// MaxScreenableToolBytes bounds the text derived from `tools_json` before it is
// handed to Cloud Model Armor, for the same reason MaxScreenableContentsBytes
// bounds the contents text: Armor's sanitize API has a request-size limit, and
// an unbounded concatenation turns a large-but-legitimate tool set into a
// screening FAILURE, which on an audit-only leg reads as "unscreened".
const MaxScreenableToolBytes = 8192

// toolTextKeys are the JSON object keys whose STRING values a model will read
// and can be induced to repeat. Everything else in a tool declaration is
// structure (types, enums, required-lists) that carries no free text.
//
// Keyed on the field name here rather than on the value, which is the opposite
// of the model-pin check, and correct for the opposite reason: there the danger
// was a value nobody declared, here the danger is free text wherever it hides,
// and the schema names the free-text fields explicitly.
var toolTextKeys = map[string]struct{}{
	"name":        {},
	"description": {},
	"title":       {},
}

// DeriveScreenableToolText extracts the model-facing free TEXT from an ADR-177
// `tools_json` payload so Armor PRE can screen the tool declarations, not just
// the prompt.
//
// WHY THIS EXISTS (CHO-2391). `tools_json` reaches chora-model-gateway from the
// caller and is screened by NOTHING. Step 5 reads req.Prompt and step 5b reads
// the text derived from `contents_json`; neither reads tool declarations. This
// was demonstrated live on the deployed build during G1'-2 acceptance: the
// prompt was benign and passed PRE with ALLOW, the injection rode in a tool
// DESCRIPTION, and the model copied it verbatim into a tool argument. The
// POST_TOOL_CALLS leg caught it on the way out, which is the only net that
// exists today, and catching it on the way out means the model has already been
// steered by it.
//
// It is not an incident today because tool declarations are platform-authored,
// so nothing learner-reachable supplies them. That bound disappears the moment
// a tenant-supplied or agent-generated tool set reaches the gateway, BYOA being
// the obvious candidate. Screening HERE rather than trusting the caller means
// no future caller can re-open the hole, which is the whole point of a
// chokepoint.
//
// WHAT IT WALKS. Both shapes callers actually send: the genai `tools` LIST and
// the bare `functionDeclarations` object. It then walks the whole structure and
// collects the string values of the free-text keys, at any depth, because a
// parameter description nests arbitrarily inside a JSON Schema and that is
// precisely where the demonstrated injection sat. Map keys are SORTED, so the
// text handed to Armor is byte-identical run to run: Go randomises map
// iteration, and an unsorted walk would make a verdict irreproducible.
//
// Returns "" for absent, malformed or text-free tools. A parse failure is NOT
// an error: this runs on the chokepoint's hot path, and a caller sending a
// shape we do not recognise must not take their turn down. The empty return is
// the signal that there was nothing to screen, and the caller decides what that
// means.
func DeriveScreenableToolText(toolsJSON string) string {
	s := strings.TrimSpace(toolsJSON)
	if s == "" {
		return ""
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return ""
	}
	var b strings.Builder
	collectToolText(v, "", &b)
	return b.String()
}

// collectToolText appends the free text of v to b, depth-first and in a stable
// order. key is the object key v was reached under ("" at the root).
func collectToolText(v any, key string, b *strings.Builder) {
	if b.Len() >= MaxScreenableToolBytes {
		return
	}
	switch t := v.(type) {
	case string:
		if _, want := toolTextKeys[key]; !want || t == "" {
			return
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		remaining := MaxScreenableToolBytes - b.Len()
		if remaining <= 0 {
			return
		}
		if len(t) > remaining {
			// Truncate on a byte bound rather than dropping the value. A
			// hostile marker is usually at the head of an oversized string;
			// dropping the whole value would screen nothing at all.
			b.WriteString(t[:remaining])
			return
		}
		b.WriteString(t)
	case []any:
		for _, item := range t {
			// A list does not rename its items, so the key carries through:
			// this keeps functionDeclarations[i].description reachable.
			collectToolText(item, key, b)
		}
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			collectToolText(t[k], k, b)
		}
	}
	// Numbers, bools and nulls carry no free text and are skipped, so a
	// non-string description contributes nothing rather than contributing
	// "123" or "<nil>".
}
