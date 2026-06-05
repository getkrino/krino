.PHONY: help lint test check fmt fmt-check clippy clean build build-release doc bench install

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint: ## Run all linting checks (fmt, clippy, tests, security)
	@./.git/hooks/pre-commit

check: ## Quick check that code compiles
	@echo "🔍 Checking compilation..."
	@cargo check --workspace

test: ## Run all tests
	@echo "🧪 Running tests..."
	@cargo test --workspace

test-lib: ## Run library tests only
	@echo "🧪 Running library tests..."
	@cargo test --lib

fmt: ## Format all code
	@echo "📝 Formatting code..."
	@cargo fmt --all

fmt-check: ## Check code formatting without modifying
	@echo "📝 Checking formatting..."
	@cargo fmt --all -- --check

clippy: ## Run clippy linter
	@echo "🔍 Running clippy..."
	@cargo clippy --workspace --all-targets --no-default-features -- -D warnings

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean

build: ## Build debug binary
	@echo "🔨 Building debug binary..."
	@cargo build --features cli

build-release: ## Build optimized release binary
	@echo "🔨 Building release binary..."
	@cargo build --release --features cli

doc: ## Generate and open documentation
	@echo "📚 Generating documentation..."
	@cargo doc --no-deps --open

doc-all: ## Generate documentation including dependencies
	@echo "📚 Generating full documentation..."
	@cargo doc --open

bench: ## Run benchmarks
	@echo "⚡ Running benchmarks..."
	@cargo bench

bench-schema: ## Run schema validation benchmarks
	@echo "⚡ Running schema validation benchmarks..."
	@cargo bench --bench schema_validation

bench-groundedness: ## Run groundedness benchmarks
	@echo "⚡ Running groundedness benchmarks..."
	@cargo bench --bench groundedness

bench-hallucination: ## Run hallucination benchmarks
	@echo "⚡ Running hallucination benchmarks..."
	@cargo bench --bench hallucination

bench-policy: ## Run policy compliance benchmarks
	@echo "⚡ Running policy compliance benchmarks..."
	@cargo bench --bench policy_compliance

install: ## Install krino CLI binary
	@echo "📦 Installing krino CLI..."
	@cargo install --path . --features cli

watch: ## Watch for changes and run checks
	@echo "👀 Watching for changes..."
	@cargo watch -x check -x test

pre-commit: ## Install git pre-commit hook
	@echo "🔗 Installing pre-commit hook..."
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Pre-commit hook installed"

ci: fmt-check clippy test ## Run all CI checks (format, clippy, tests)
	@echo "✅ All CI checks passed!"
