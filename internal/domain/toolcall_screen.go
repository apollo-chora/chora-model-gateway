package domain

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// maxToolCallWalkDepth bounds how deep the argument walk recurses.
//
// This runs on the chokepoint's hot path against a payload the MODEL produced,
// so unbounded recursion is a denial-of-service surface rather than a
// theoretical concern. No legitimate tool argument nests thirty-two levels; a
// payload that does is skipped below the cap rather than walked, which loses
// nothing real and cannot exhaust the stack.
const maxToolCallWalkDepth = 32

// DeriveScreenableToolCallText extracts the model-facing TEXT from an ADR-177
// `tool_calls_json` payload so Armor POST can screen what the model actually
// emitted, rather than skipping the turn because the completion was empty.
//
// WHY THIS EXISTS (G1'-2, the tool-call half). service.go treats a turn with no
// completion text as having nothing to screen, and for an IMAGE-only response
// that is correct: Armor is a text guardrail and there is no text. For a
// TOOL-CALL-only response it is a hole. A functionCall carries model-emitted
// text in its arguments, and that text is an egress channel: a prompt-injected
// model that cannot say a payload in its completion can still put it in a tool
// argument, and nothing looked at it.
//
// WHAT IT DELIBERATELY DOES NOT DO. Non-string scalars (numbers, booleans,
// null) are skipped, for the same reason `DeriveScreenableText` skips
// inlineData: Armor is a text guardrail, and rendering a number into the
// screened text is noise rather than safety. The tool NAME is emitted only
// alongside at least one string leaf, because the name comes from the declared
// tool set rather than from the model's free text: on its own it is not content
// worth screening, and screening it alone would spend an Armor call per turn to
// re-read a constant.
//
// The rendering is plain `path: value` lines rather than the raw JSON. Armor
// reads text, and JSON punctuation is noise that dilutes what the detector is
// looking at; the path prefix keeps the argument legible ("to: x@y" reads as a
// recipient, "x@y" alone does not).
//
// Map keys are SORTED. Go map iteration order is random, so an unsorted walk
// would hand Armor a differently-ordered string on every call, making a verdict
// irreproducible and an evidence row unexplainable.
//
// Returns "" for absent, malformed or text-free tool calls. A parse failure is
// NOT an error: a caller sending a shape we do not recognise must not take
// their turn down. The empty return is the signal that there was nothing to
// screen, and the caller decides what that means.
func DeriveScreenableToolCallText(toolCallsJSON string) string {
	s := strings.TrimSpace(toolCallsJSON)
	if s == "" {
		return ""
	}

	// The wire shape is what gemini.go marshals: a JSON array of
	// {"name":..., "args":{...}}. Args stays raw so a non-object `args` is
	// skipped rather than failing the whole payload.
	var calls []struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	}
	if err := json.Unmarshal([]byte(s), &calls); err != nil {
		return ""
	}

	var b strings.Builder
	for _, c := range calls {
		if len(c.Args) == 0 {
			continue
		}
		var args map[string]any
		if err := json.Unmarshal(c.Args, &args); err != nil {
			continue
		}
		var leaves []string
		collectStringLeaves("", args, &leaves, 0)
		if len(leaves) == 0 {
			continue
		}
		if c.Name != "" {
			appendScreenLine(&b, "tool: "+c.Name)
		}
		for _, leaf := range leaves {
			appendScreenLine(&b, leaf)
			if b.Len() >= MaxScreenableContentsBytes {
				return truncateUTF8(b.String(), MaxScreenableContentsBytes)
			}
		}
	}
	return truncateUTF8(b.String(), MaxScreenableContentsBytes)
}

// collectStringLeaves walks an argument value and appends one `path: value`
// entry per non-empty STRING leaf, in a deterministic order.
func collectStringLeaves(path string, v any, out *[]string, depth int) {
	if depth > maxToolCallWalkDepth {
		return
	}
	switch t := v.(type) {
	case string:
		if t == "" {
			return
		}
		if path == "" {
			*out = append(*out, t)
			return
		}
		*out = append(*out, path+": "+t)
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			collectStringLeaves(joinArgPath(path, k), t[k], out, depth+1)
		}
	case []any:
		for i, e := range t {
			collectStringLeaves(path+"["+strconv.Itoa(i)+"]", e, out, depth+1)
		}
	}
	// Numbers, booleans and null carry no screenable text and are skipped.
}

func joinArgPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

func appendScreenLine(b *strings.Builder, line string) {
	if b.Len() > 0 {
		b.WriteByte('\n')
	}
	b.WriteString(line)
}
