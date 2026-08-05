# Stage 1: build the API binary.
FROM rust:1.93-bookworm AS builder

WORKDIR /build
COPY . .

RUN cargo build --release --package krino-api

# Stage 2: minimal runtime.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/krino-api /usr/local/bin/krino-api

# Mount the models directory and config at runtime — they aren't baked in.
# Example:
#   docker run \
#     -v $(pwd)/models:/opt/krino/models:ro \
#     -v $(pwd)/krino-api.toml:/opt/krino/krino-api.toml:ro \
#     -p 8080:8080 \
#     krino:latest
WORKDIR /opt/krino

RUN useradd -r -s /bin/false krino && \
    mkdir -p /opt/krino/models && \
    chown -R krino:krino /opt/krino
USER krino

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["krino-api"]
