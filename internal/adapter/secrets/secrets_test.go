package secrets_test

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/secrets"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubTokens struct {
	token string
	err   error
}

func (s stubTokens) Token(ctx context.Context) (string, error) { return s.token, s.err }

func newClient(t *testing.T, srv *httptest.Server) *secrets.Client {
	t.Helper()
	c, err := secrets.New(secrets.Config{
		HTTPClient:  srv.Client(),
		Tokens:      stubTokens{token: "test-bearer"},
		Project:     "chora-489812",
		Environment: "dev",
		Endpoint:    srv.URL,
	})
	require.NoError(t, err)
	return c
}

func enc(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func TestResolveByoaKey_Found(t *testing.T) {
	var capturedPath, capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"payload":{"data":"` + enc("sk-tenant-byoa-key") + `"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	key, err := c.ResolveByoaKey(context.Background(), "mighty-mind-tuition", domain.VendorFamilyOpenAI)
	require.NoError(t, err)
	assert.Equal(t, "sk-tenant-byoa-key", key)
	assert.Equal(t, "Bearer test-bearer", capturedAuth)
	// Path matches the BYOA naming convention.
	assert.Contains(t, capturedPath, "/projects/chora-489812/secrets/chora-byoa-mighty-mind-tuition-openai_byoa-api-key/versions/latest:access")
}

func TestResolveByoaKey_NotFound_ReturnsEmptyNoError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	key, err := c.ResolveByoaKey(context.Background(), "no-byoa-tenant", domain.VendorFamilyOpenAI)
	require.NoError(t, err)
	assert.Equal(t, "", key)
}

func TestResolveByoaKey_OtherHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":"PERMISSION_DENIED"}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveByoaKey(context.Background(), "tenant-1", domain.VendorFamilyOpenAI)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "403")
	assert.Contains(t, err.Error(), "PERMISSION_DENIED")
}

func TestResolveGlobalKey_Found(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_, _ = w.Write([]byte(`{"payload":{"data":"` + enc("sk-global-platform-key") + `"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	key, err := c.ResolveGlobalKey(context.Background(), domain.VendorFamilyOpenAI)
	require.NoError(t, err)
	assert.Equal(t, "sk-global-platform-key", key)
	// Vendor-key naming includes environment suffix.
	assert.Contains(t, capturedPath, "/secrets/chora-vendor-openai_byoa-api-key-dev/versions/latest:access")
}

func TestResolveByoaKey_TokenError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c, err := secrets.New(secrets.Config{
		HTTPClient:  srv.Client(),
		Tokens:      stubTokens{err: errors.New("WIF mint failed")},
		Project:     "chora-489812",
		Environment: "dev",
		Endpoint:    srv.URL,
	})
	require.NoError(t, err)
	_, err = c.ResolveByoaKey(context.Background(), "tenant-1", domain.VendorFamilyAnthropic)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "WIF mint failed")
}

func TestResolveByoaKey_InvalidBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"payload":{"data":"!!!not-base64!!!"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveByoaKey(context.Background(), "t", domain.VendorFamilyOpenAI)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "base64")
}

// ResolveSecret has FAIL-LOUD semantics distinct from the BYOA + global
// resolvers — 404 must surface as an error, not silently as "".

func TestResolveSecret_Found(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_, _ = w.Write([]byte(`{"payload":{"data":"` + enc("postgres://u:p@127.0.0.1:5432/chora_observability?sslmode=require") + `"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	dsn, err := c.ResolveSecret(context.Background(), "chora-dev-cloudsql-chora_observability-app_rw-dsn")
	require.NoError(t, err)
	assert.Equal(t, "postgres://u:p@127.0.0.1:5432/chora_observability?sslmode=require", dsn)
	assert.Contains(t, capturedPath, "/projects/chora-489812/secrets/chora-dev-cloudsql-chora_observability-app_rw-dsn/versions/latest:access")
}

func TestResolveSecret_NotFound_FailsLoud(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveSecret(context.Background(), "chora-dev-cloudsql-chora_observability-app_rw-dsn")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_FOUND")
}

func TestResolveSecret_EmptyPayload_FailsLoud(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Valid response shape with an empty payload — still fail-loud
		// because a required boot-time secret cannot be empty.
		_, _ = w.Write([]byte(`{"payload":{"data":""}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveSecret(context.Background(), "chora-dev-cloudsql-chora_observability-app_rw-dsn")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_FOUND")
}

func TestResolveSecret_EmptySecretID_Rejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveSecret(context.Background(), "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "secretID required")
}

func TestResolveSecret_OtherHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":"PERMISSION_DENIED"}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveSecret(context.Background(), "chora-dev-cloudsql-chora_observability-app_rw-dsn")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "403")
	assert.Contains(t, err.Error(), "PERMISSION_DENIED")
}

func TestResolveByoaKey_PayloadTrailingNewline(t *testing.T) {
	// `gcloud secrets versions add --data-file=` often appends a newline;
	// the adapter must strip it so vendor adapters get a clean key.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"payload":{"data":"` + enc("sk-with-newline\n") + `"}}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	key, err := c.ResolveByoaKey(context.Background(), "t", domain.VendorFamilyOpenAI)
	require.NoError(t, err)
	assert.Equal(t, "sk-with-newline", key)
	assert.False(t, strings.HasSuffix(key, "\n"))
}

func TestResolveByoaKey_InvalidResponseJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.ResolveByoaKey(context.Background(), "t", domain.VendorFamilyOpenAI)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unmarshal")
}

func TestNew_Validation(t *testing.T) {
	_, err := secrets.New(secrets.Config{Project: "p", Environment: "dev"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "TokenProvider")

	_, err = secrets.New(secrets.Config{Tokens: stubTokens{}, Environment: "dev"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Project")

	_, err = secrets.New(secrets.Config{Tokens: stubTokens{}, Project: "p"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Environment")
}
