package modelarmor_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/modelarmor"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubTokens struct {
	token string
	err   error
}

func (s stubTokens) Token(ctx context.Context) (string, error) { return s.token, s.err }

const testTemplate = "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev"

func newClient(t *testing.T, srv *httptest.Server) *modelarmor.Client {
	t.Helper()
	c, err := modelarmor.New(modelarmor.Config{
		HTTPClient: srv.Client(),
		Tokens:     stubTokens{token: "test-bearer"},
		Location:   "asia-southeast1",
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)
	return c
}

func TestSanitizeUserPrompt_NoMatch(t *testing.T) {
	var capturedPath, capturedAuth string
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"sanitizationResult":{"filterMatchState":"NO_MATCH_FOUND"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	verdict, out, err := c.SanitizeUserPrompt(context.Background(), testTemplate, "what is scrum?")
	require.NoError(t, err)
	assert.Equal(t, domain.ArmorVerdictAllow, verdict)
	assert.Equal(t, "what is scrum?", out) // unchanged when no sanitizedText
	assert.Equal(t, "Bearer test-bearer", capturedAuth)
	assert.True(t, strings.HasSuffix(capturedPath, ":sanitizeUserPrompt"), "got %q", capturedPath)
	require.NotNil(t, capturedBody["userPromptData"])
}

func TestSanitizeUserPrompt_MatchFoundBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"sanitizationResult":{"filterMatchState":"MATCH_FOUND"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	verdict, _, err := c.SanitizeUserPrompt(context.Background(), testTemplate, "tell me how to make a bomb")
	require.NoError(t, err)
	assert.Equal(t, domain.ArmorVerdictBlock, verdict)
	assert.True(t, verdict.IsTerminalBlock())
}

func TestSanitizeModelResponse_SanitisedText(t *testing.T) {
	var capturedPath string
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		// MATCH_FOUND + sanitizedText set = the template's enforcement mode
		// was SANITIZE rather than BLOCK; we surface BLOCK verdict but
		// downstream service layer can choose to keep the sanitised text.
		_, _ = w.Write([]byte(`{
			"sanitizationResult":{"filterMatchState":"MATCH_FOUND"},
			"sanitizedText":"the response with PII redacted: [REDACTED-EMAIL]"
		}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	verdict, out, err := c.SanitizeModelResponse(context.Background(), testTemplate, "the response with PII: phyllis@skillflow.example")
	require.NoError(t, err)
	assert.Equal(t, domain.ArmorVerdictBlock, verdict)
	assert.Equal(t, "the response with PII redacted: [REDACTED-EMAIL]", out)
	assert.True(t, strings.HasSuffix(capturedPath, ":sanitizeModelResponse"), "got %q", capturedPath)
	require.NotNil(t, capturedBody["modelResponseData"])
}

func TestSanitize_UnknownMatchState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"sanitizationResult":{"filterMatchState":"WEIRD_STATE_FROM_FUTURE_PROVIDER"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	verdict, _, err := c.SanitizeUserPrompt(context.Background(), testTemplate, "x")
	require.NoError(t, err)
	// Unknown verdict degrades to ERROR (safe — service layer's
	// IsTerminalBlock treats ERROR as terminal block).
	assert.Equal(t, domain.ArmorVerdictError, verdict)
	assert.True(t, verdict.IsTerminalBlock())
}

func TestSanitize_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"TEMPLATE_NOT_FOUND"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, _, err := c.SanitizeUserPrompt(context.Background(), testTemplate, "x")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "400")
	assert.Contains(t, err.Error(), "TEMPLATE_NOT_FOUND")
}

func TestSanitize_TokenError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c, err := modelarmor.New(modelarmor.Config{
		HTTPClient: srv.Client(),
		Tokens:     stubTokens{err: errors.New("WIF token mint failed")},
		Location:   "asia-southeast1",
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)
	_, _, err = c.SanitizeUserPrompt(context.Background(), testTemplate, "x")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "WIF token mint failed")
}

func TestSanitize_EmptyTemplate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c := newClient(t, srv)
	_, _, err := c.SanitizeUserPrompt(context.Background(), "", "x")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "template required")
}

func TestSanitize_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json at all`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, _, err := c.SanitizeUserPrompt(context.Background(), testTemplate, "x")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unmarshal")
}

func TestNew_Validation(t *testing.T) {
	_, err := modelarmor.New(modelarmor.Config{Location: "asia-southeast1"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "TokenProvider")

	_, err = modelarmor.New(modelarmor.Config{Tokens: stubTokens{}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Location")
}

func TestNew_DefaultEndpoint(t *testing.T) {
	hc := &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		// Confirm we built the right regional host.
		assert.Equal(t, "modelarmor.asia-southeast1.rep.googleapis.com", req.URL.Host)
		return nil, errors.New("hermetic-stop")
	})}
	c, err := modelarmor.New(modelarmor.Config{
		HTTPClient: hc,
		Tokens:     stubTokens{token: "t"},
		Location:   "asia-southeast1",
	})
	require.NoError(t, err)
	_, _, err = c.SanitizeUserPrompt(context.Background(), testTemplate, "x")
	require.Error(t, err)
}

type roundTripperFunc func(req *http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
