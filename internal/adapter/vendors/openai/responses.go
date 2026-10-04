package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// This file implements the OpenAI **Responses** API surface — the one the
// OpenAI-compatible aggregators (Meta's api.meta.ai among them) expose for
// hosted web search:
//
//	POST {base_url}/responses
//	{"model": "...", "input": "...", "tools": [{"type": "web_search"}]}
//
// The Responses API differs from chat completions in three ways that matter
// here:
//
//   - the prompt field is `input`, not `messages`
//   - the reply is a flat `output` array of typed items, NOT `choices`
//   - citations arrive as `url_citation` annotations on the output text
//
// The output array must be parsed BY TYPE, never by position: a reasoning model
// interleaves `reasoning` items with `web_search_call` and `message`, and
// indexing into it is how a grounded answer turns into an empty one.

// buildResponsesBody assembles the request for a grounded call.
func buildResponsesBody(req domain.VendorRequest) map[string]any {
	body := map[string]any{
		"model":  req.Target.UpstreamModel,
		"stream": false,
	}

	// `input` takes either a bare string or an array of input items. A
	// multi-turn conversation is already in item form, so it forwards as-is;
	// a single prompt collapses to the string form, which is what the docs
	// show and what every implementation definitely parses.
	if req.ContentsJSON != "" {
		var items []json.RawMessage
		if err := json.Unmarshal([]byte(req.ContentsJSON), &items); err == nil && len(items) > 0 {
			body["input"] = json.RawMessage(req.ContentsJSON)
		} else {
			body["input"] = req.Prompt
		}
	} else {
		body["input"] = req.Prompt
	}

	if req.SystemPrompt != "" {
		// The Responses API has an `instructions` field for the system prompt
		// rather than a system message in the input array.
		body["instructions"] = req.SystemPrompt
	}

	body["tools"] = buildGroundingTools(req)

	// Generation knobs are forwarded the same way as on chat completions, so a
	// self-hosted server's extensions still pass through.
	for k, v := range req.GenerationConfig {
		switch k {
		case "temperature", "top_p", "max_output_tokens", "max_tokens", "seed", "reasoning_effort", "parallel_tool_calls":
			body[k] = v
		}
	}
	return body
}

// buildGroundingTools renders the web-search tool in whatever shape the
// configured provider expects.
//
// The tool is built from the registry's Grounding block, never from the
// caller's JSON: a caller may ASK for grounding, but what actually goes on the
// wire is the operator's configuration.
func buildGroundingTools(req domain.VendorRequest) []any {
	g := req.Target.Grounding
	if g == nil {
		return nil
	}
	surface := g.EffectiveSurface(req.Target.Vendor)

	tool := map[string]any{"type": g.EffectiveToolType(surface)}

	switch surface {
	case domain.SurfaceMessages:
		// Anthropic's server tools require a `name`. Not reachable through
		// this adapter, but the shape is cheap to honour and keeps the
		// registry honest about what it configures.
		tool["name"] = g.EffectiveToolName()
		if g.MaxUses > 0 {
			tool["max_uses"] = g.MaxUses
		}
	case domain.SurfaceChatCompletions:
		// Chat completions has no separate name/max_uses convention.
	default:
		// The Responses API tool takes provider options inline.
		if g.MaxUses > 0 {
			tool["max_uses"] = g.MaxUses
		}
	}

	for k, v := range g.ExtraToolFields {
		// The registry's extra fields win over nothing and lose to the
		// computed discriminator, so a stray `type:` cannot break the tool.
		if k == "type" {
			continue
		}
		tool[k] = v
	}
	return []any{tool}
}

// responsesReply is the subset of the Responses payload the gateway reads.
type responsesReply struct {
	ID     string          `json:"id"`
	Model  string          `json:"model"`
	Status string          `json:"status"`
	Output []responsesItem `json:"output"`

	// IncompleteDetails explains a `status: incomplete` reply — typically
	// `{"reason": "max_output_tokens"}` when a multi-step research run ran out
	// of budget mid-search. It is the difference between "the model answered"
	// and "the model was cut off", and it is NOT in `error`.
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`

	// Error carries a provider-side failure on an otherwise-200 response.
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`

	Usage struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
		InputDetails struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}

// responsesItem is one element of the output array. The items are heterogeneous,
// so this is a union decoded by the `type` discriminator.
type responsesItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Status string `json:"status"`
	Role   string `json:"role"`

	// A web_search_call carries what was actually searched.
	Action struct {
		Type   string   `json:"type"`
		Query  string   `json:"query"`
		Querys []string `json:"queries"`
	} `json:"action"`

	// A message carries content parts.
	Content []responsesContent `json:"content"`
}

type responsesContent struct {
	Type string `json:"type"`
	Text string `json:"text"`

	Annotations []responsesAnnotation `json:"annotations"`
}

type responsesAnnotation struct {
	Type       string `json:"type"`
	URL        string `json:"url"`
	Title      string `json:"title"`
	StartIndex *int   `json:"start_index"`
	EndIndex   *int   `json:"end_index"`
}

// generateGrounded dispatches a call through the Responses API and normalises
// the reply.
func (c *Client) generateGrounded(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	raw, err := json.Marshal(buildResponsesBody(req))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	var parsed responsesReply
	if err := c.do(ctx, req, req.Target.GroundingURL(), raw, &parsed); err != nil {
		return domain.VendorResponse{}, err
	}
	return c.responsesToDomain(req, parsed), nil
}

func (c *Client) responsesToDomain(req domain.VendorRequest, r responsesReply) domain.VendorResponse {
	out := domain.VendorResponse{
		ModelVersion: firstNonEmpty(r.Model, req.Target.UpstreamModel),
		FinishReason: domain.FinishReasonComplete,
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			CachedTokens: r.Usage.InputDetails.CachedTokens,
			CostMicros: c.costMicros(req.Target.LogicalModelID,
				r.Usage.InputTokens, r.Usage.OutputTokens, r.Usage.InputDetails.CachedTokens),
		},
	}

	// The terminal status decides the finish reason, and it is checked BEFORE
	// the output is read. A multi-step grounded run that exhausts its token
	// budget returns `status: "incomplete"` with a handful of intermediate
	// narration messages and no answer — reporting that as a successful
	// completion hands the caller a confident-sounding non-answer.
	switch r.Status {
	case "", "completed":
		// The common case; nothing to add.
	case "incomplete":
		reason := "the provider stopped before finishing"
		if r.IncompleteDetails != nil && r.IncompleteDetails.Reason != "" {
			reason = r.IncompleteDetails.Reason
		}
		out.FinishReason = domain.FinishReasonMaxTokens
		out.FinishDetail = "the response was truncated (" + reason +
			"); a grounded run needs enough max_output_tokens to finish its searches"
	case "failed":
		if r.Error != nil && r.Error.Message != "" {
			out.FinishDetail = "the provider reported a failure: " + r.Error.Message
		} else {
			out.FinishDetail = "the provider reported a failure"
		}
		out.FinishReason = domain.FinishReasonVendorError
	case "cancelled":
		out.FinishReason = domain.FinishReasonVendorError
		out.FinishDetail = "the response was cancelled upstream"
	default:
		out.FinishReason = domain.FinishReasonUnspecified
		out.FinishDetail = "the response finished in status " + r.Status
	}

	// Messages are kept SEPARATE rather than concatenated. A reasoning model
	// narrates between its searches and states the answer last; joining them
	// yields narration soup whose final line is a half-finished sentence and no
	// answer at all — observed live against api.meta.ai 2026-10-04. Within one
	// message item the output_text parts ARE one answer, so those are joined.
	for _, item := range r.Output {
		// Parse by type, never by position: a reasoning model interleaves
		// `reasoning` items with the search calls and the messages.
		switch item.Type {
		case "web_search_call":
			out.SearchQueries = append(out.SearchQueries, searchQueriesFrom(item)...)

		case "message":
			var b strings.Builder
			var citations []domain.GroundingCitation
			for _, part := range item.Content {
				if part.Type != "output_text" && part.Type != "" && part.Text == "" {
					continue
				}
				b.WriteString(part.Text)
				citations = append(citations, citationsFromAnnotations(part.Annotations)...)
			}
			if b.Len() == 0 && len(citations) == 0 {
				continue
			}
			out.Messages = append(out.Messages, domain.GroundingMessage{
				Text:      b.String(),
				Citations: citations,
			})
			out.Citations = append(out.Citations, citations...)
		}
	}

	// The answer is the LAST message — the one that carries the citations.
	if len(out.Messages) > 0 {
		out.Completion = out.Messages[len(out.Messages)-1].Text
	}

	// An incomplete run usually has SOME text — the model's intermediate
	// reasoning-as-prose — and no answer. Reporting an empty completion would
	// hide the partial result, so it is returned alongside the failure.
	if out.FinishReason == domain.FinishReasonComplete && out.Completion == "" {
		out.FinishReason = domain.FinishReasonUnspecified
		out.FinishDetail = "the provider returned no output_text content"
	}
	return out
}

// searchQueriesFrom extracts the queries a search call actually ran.
//
// The action is NOT always a search: a reasoning model also emits
// `{"type":"open_page","url":"..."}` once it decides to read a result. Those
// are navigation, not queries, and reporting a page URL as a search query
// would mislead whoever debugs an answer. Only `search` actions are collected.
func searchQueriesFrom(item responsesItem) []string {
	queries := make([]string, 0, 2)
	if item.Action.Type == "search" || item.Action.Type == "" {
		if item.Action.Query != "" {
			queries = append(queries, item.Action.Query)
		}
		queries = append(queries, item.Action.Querys...)
	}
	return queries
}

// citationsFromAnnotations keeps only the url_citation annotations. The array
// can also carry file_citation entries when file_search is in play, and those
// are not web sources, so they are dropped rather than mislabelled.
func citationsFromAnnotations(annotations []responsesAnnotation) []domain.GroundingCitation {
	var out []domain.GroundingCitation
	for _, a := range annotations {
		if a.URL == "" {
			continue
		}
		if a.Type != "" && a.Type != "url_citation" {
			continue
		}
		c := domain.GroundingCitation{URL: a.URL, Title: a.Title}
		if a.StartIndex != nil {
			c.StartIndex = *a.StartIndex
		}
		if a.EndIndex != nil {
			c.EndIndex = *a.EndIndex
		}
		out = append(out, c)
	}
	return out
}
