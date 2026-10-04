module github.com/apollo-chora/chora-model-gateway

go 1.26.1

require (
	github.com/google/uuid v1.6.0
	github.com/jackc/pgx/v5 v5.9.2
	github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go v0.0.0-00010101000000-000000000000
	go.opentelemetry.io/otel/trace v1.44.0
	google.golang.org/grpc v1.83.0
	google.golang.org/protobuf v1.36.11
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/rogpeppe/go-internal v1.16.0 // indirect
	go.opentelemetry.io/otel v1.44.0 // indirect
	golang.org/x/net v0.57.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260615183401-62b3387ff324 // indirect
)

// The generated gRPC stubs for the ModelGatewayService contract live in ./gen
// in this repository.
//
// Upstream they are generated into a separate monorepo module
// (chora-contracts) and consumed via a `replace` pointing outside this tree.
// Vendoring them here is what makes this repository self-contained, so a fresh
// clone builds and tests with nothing else checked out.
//
// TO RE-SYNC after a .proto change:
//
//	cp <monorepo>/chora-contracts/gen/go/chora/services/model_gateway/v1/*.go \
//	   gen/chora/services/model_gateway/v1/
//	go mod tidy && go build ./...
//
// A stale copy here is a SILENT wire-format drift, not a build error: the
// service will still compile and simply speak an older protocol.
replace github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go => ./gen
