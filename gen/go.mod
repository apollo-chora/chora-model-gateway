// Vendored subset of the chora-contractts generated stubs.
//
// This module exists so chora-model-gateway can be built and tested as a
// STANDALONE repository. Upstream, ModelGatewayService's stubs are generated
// into the monorepo's chora-contracts module and consumed via a `replace`
// directive pointing at ../../chora-contracts/gen/go.
//
// Only the ONE package this service imports is vendored — the gRPC surface.
// The file also documents how to re-sync it, because a stale copy of a proto
// contract is a silent wire-format drift rather than a build error.
module github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go

go 1.26.1

require (
	google.golang.org/grpc v1.83.0
	google.golang.org/protobuf v1.36.11
)

require (
	go.opentelemetry.io/otel v1.44.0 // indirect
	go.opentelemetry.io/otel/sdk/metric v1.44.0 // indirect
	golang.org/x/crypto v0.55.0 // indirect
	golang.org/x/net v0.57.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260615183401-62b3387ff324 // indirect
)
