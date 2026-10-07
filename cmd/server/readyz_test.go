// readyz_test.go — the DB-backed readiness probe handler (Python parity:
// app/http.py ready() → database.ping()). The handler is pure HTTP + one
// Ping call, so a minimal fake database/sql driver suffices — no real
// Postgres needed in the unit-test stage.
package main

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
)

// errPingFailed simulates an unreachable database.
var errPingFailed = errors.New("connection refused")

// pingDriver is a minimal database/sql driver that only implements Ping —
// enough to exercise handleReadyz without a real Postgres.
type pingDriver struct{ err error }

func (d pingDriver) Open(string) (driver.Conn, error) { return pingConn{err: d.err}, nil }

type pingConn struct{ err error }

func (c pingConn) Prepare(string) (driver.Stmt, error) {
	return nil, errors.New("pingConn: prepare not implemented")
}
func (c pingConn) Close() error { return nil }
func (c pingConn) Begin() (driver.Tx, error) {
	return nil, errors.New("pingConn: begin not implemented")
}
func (c pingConn) Ping(context.Context) error { return c.err }

func init() {
	sql.Register("readyz-ping-ok", pingDriver{})
	sql.Register("readyz-ping-fail", pingDriver{err: errPingFailed})
}

func newPingRepo(t *testing.T, pingErr error) *pg.Repo {
	t.Helper()
	name := "readyz-ping-ok"
	if pingErr != nil {
		name = "readyz-ping-fail"
	}
	db, err := sql.Open(name, "")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	repo, err := pg.New(db)
	if err != nil {
		t.Fatalf("pg.New: %v", err)
	}
	return repo
}

func TestHandleReadyz_Ready(t *testing.T) {
	repo := newPingRepo(t, nil)
	rec := httptest.NewRecorder()
	handleReadyz(repo).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if rec.Body.String() != "ready" {
		t.Fatalf("body = %q, want %q", rec.Body.String(), "ready")
	}
}

func TestHandleReadyz_NotReady(t *testing.T) {
	repo := newPingRepo(t, errPingFailed)
	rec := httptest.NewRecorder()
	handleReadyz(repo).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if rec.Body.String() != "not ready" {
		t.Fatalf("body = %q, want %q", rec.Body.String(), "not ready")
	}
}
