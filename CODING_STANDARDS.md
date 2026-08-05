# Coding Standards for Krino

This document defines the coding standards for the Krino project. All code contributions must adhere to these standards.

## References

This document is based on:
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/about.html)
- [Microsoft Rust Guidelines](https://microsoft.github.io/rust-guidelines/)

## Table of Contents

1. [Rust Standards](#rust-standards)
2. [Python Standards](#python-standards)
3. [General Principles](#general-principles)

---

## Rust Standards

### Naming Conventions (C-CASE)

- **Crates, modules, functions, methods**: `snake_case`
  - ✅ `arachne_worker`, `fetch_url`, `process_task`
  - ❌ `ArachneWorker`, `fetchURL`, `ProcessTask`

- **Types, traits, enums, enum variants**: `PascalCase`
  - ✅ `FetchMode`, `TaskResult`, `BrowserPool`
  - ❌ `fetch_mode`, `task_result`, `browser_pool`

- **Constants, statics**: `SCREAMING_SNAKE_CASE`
  - ✅ `MAX_RETRIES`, `DEFAULT_TIMEOUT`
  - ❌ `maxRetries`, `defaultTimeout`

- **Type parameters**: Single uppercase letter or `PascalCase`
  - ✅ `T`, `E`, `Iterator`, `TcpStream`
  - ❌ `t`, `iter`, `tcp_stream`

### Error Handling (C-FAILURE)

- **Use `Result<T, E>` for fallible operations**
  ```rust
  // ✅ Good
  pub async fn fetch_url(url: &str) -> Result<Response, FetchError> {
      // ...
  }

  // ❌ Bad - panicking in library code
  pub async fn fetch_url(url: &str) -> Response {
      let resp = client.get(url).send().await.unwrap();
      resp
  }
  ```

- **Never `unwrap()` or `expect()` in production code**
  - Use `?` operator for propagation
  - Handle `Option` explicitly with `if let` or `match`
  - Only use `unwrap()` in tests or where mathematically impossible to fail

- **Use `anyhow::Result` in application code**
  ```rust
  // In arachne-worker/src/
  use anyhow::{Context, Result};

  async fn process_task(task: &Task) -> Result<()> {
      let response = fetch_url(&task.url)
          .await
          .context("Failed to fetch URL")?;
      Ok(())
  }
  ```

- **Use `thiserror` for library errors**
  ```rust
  // In arachne-common/src/error.rs
  use thiserror::Error;

  #[derive(Error, Debug)]
  pub enum FetchError {
      #[error("HTTP request failed: {0}")]
      HttpError(#[from] reqwest::Error),

      #[error("Invalid URL: {0}")]
      InvalidUrl(String),

      #[error("Rate limit exceeded for domain: {domain}")]
      RateLimitExceeded { domain: String },
  }
  ```

### Documentation (C-DOCS)

- **All public items must have documentation comments**
  ```rust
  /// Fetches a URL using the configured HTTP client.
  ///
  /// # Arguments
  ///
  /// * `url` - The URL to fetch
  /// * `client` - The HTTP client to use
  ///
  /// # Errors
  ///
  /// Returns `FetchError::HttpError` if the request fails
  /// Returns `FetchError::InvalidUrl` if the URL is malformed
  ///
  /// # Examples
  ///
  /// ```no_run
  /// use arachne_worker::fetch_url;
  ///
  /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
  /// let response = fetch_url("https://example.com").await?;
  /// # Ok(())
  /// # }
  /// ```
  pub async fn fetch_url(url: &str, client: &Client) -> Result<Response, FetchError> {
      // ...
  }
  ```

- **Document error conditions in `# Errors` section**
- **Document panics (if any) in `# Panics` section**
- **Document safety requirements in `# Safety` section for `unsafe` code**

### Async/Concurrency (C-ASYNC)

- **Use `tokio` runtime exclusively**
  - No `async-std` or other runtimes
  - Use `#[tokio::main]` for main functions
  - Use `#[tokio::test]` for async tests

- **Use structured concurrency**
  ```rust
  // ✅ Good - bounded concurrency with proper error handling
  use tokio::task::JoinSet;

  async fn process_batch(tasks: Vec<Task>) -> Result<Vec<TaskResult>> {
      let mut set = JoinSet::new();

      for task in tasks {
          set.spawn(async move {
              process_task(&task).await
          });
      }

      let mut results = Vec::new();
      while let Some(res) = set.join_next().await {
          results.push(res??);
      }
      Ok(results)
  }

  // ❌ Bad - unbounded spawning
  async fn process_batch(tasks: Vec<Task>) {
      for task in tasks {
          tokio::spawn(async move {
              process_task(&task).await
          });
      }
      // No way to collect results or handle errors!
  }
  ```

- **Use `tokio::select!` for cancellation**
  ```rust
  use tokio::signal;

  loop {
      tokio::select! {
          _ = signal::ctrl_c() => {
              info!("Shutdown signal received");
              break;
          }
          result = worker.process_next() => {
              handle_result(result)?;
          }
      }
  }
  ```

### Logging and Tracing (C-OBSERVABILITY)

- **Use `tracing` crate, never `log` or `println!`**
  ```rust
  use tracing::{info, warn, error, debug, instrument};

  #[instrument(skip(client), fields(task_id = %task.id))]
  async fn process_task(task: &Task, client: &Client) -> Result<()> {
      info!("Starting task processing");

      match fetch_url(&task.url, client).await {
          Ok(response) => {
              debug!(status = response.status().as_u16(), "Fetch succeeded");
              Ok(())
          }
          Err(e) => {
              error!(error = %e, "Fetch failed");
              Err(e.into())
          }
      }
  }
  ```

- **Use `#[instrument]` on key functions**
- **Include context in log messages**: `task_id`, `domain`, `worker_id`
- **Use structured fields, not string interpolation**
  ```rust
  // ✅ Good
  info!(task_id = %task.id, domain = %task.domain, "Task completed");

  // ❌ Bad
  info!("Task {} completed for domain {}", task.id, task.domain);
  ```

### Type Safety (C-TYPES)

- **Use newtype pattern for domain concepts**
  ```rust
  // ✅ Good
  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct TaskId(String);

  impl TaskId {
      pub fn new(campaign_id: &str, url: &str) -> Self {
          let hash = sha256(format!("{}{}", campaign_id, url));
          TaskId(hash)
      }
  }

  // ❌ Bad - primitive obsession
  pub type TaskId = String;
  ```

- **Use enums for state machines**
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum FetchMode {
      Static,
      Browser,
      Auto,
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub enum TaskStatus {
      Pending,
      Running { worker_id: String, started_at: Instant },
      Succeeded { duration: Duration },
      Failed { error: String, retry_count: u32 },
  }
  ```

- **Prefer `&str` over `&String` in function parameters**
- **Prefer `impl Trait` or generics over `Box<dyn Trait>` where possible**

### Resource Management (C-RESOURCES)

- **Implement `Drop` for cleanup**
  ```rust
  pub struct BrowserPool {
      browser: Browser,
      active_contexts: Vec<BrowserContext>,
  }

  impl Drop for BrowserPool {
      fn drop(&mut self) {
          // Cleanup browser contexts
          for context in &self.active_contexts {
              let _ = context.close();
          }
      }
  }
  ```

- **Use RAII for locks and leases**
  ```rust
  pub struct SemaphoreLease {
      domain: String,
      worker_id: String,
      redis: RedisClient,
  }

  impl Drop for SemaphoreLease {
      fn drop(&mut self) {
          // Release lease on drop
          let _ = self.redis.release_semaphore(&self.domain, &self.worker_id);
      }
  }
  ```

### Testing (C-TESTING)

- **Unit tests in same file as implementation**
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_task_id_generation() {
          let id1 = TaskId::new("campaign1", "https://example.com");
          let id2 = TaskId::new("campaign1", "https://example.com");
          assert_eq!(id1, id2, "Same inputs should produce same TaskId");
      }

      #[tokio::test]
      async fn test_fetch_url_success() {
          let client = build_test_client();
          let result = fetch_url("https://example.com", &client).await;
          assert!(result.is_ok());
      }
  }
  ```

- **Integration tests in `tests/` directory**
- **Use `testcontainers` for Redis, PostgreSQL tests**
- **Use `#[tokio::test]` for async tests**
- **Use descriptive test names**: `test_<what>_<condition>_<expected>`

### Performance (C-PERFORMANCE)

- **Avoid cloning large data structures**
  ```rust
  // ✅ Good - borrow
  fn process_task(task: &Task) -> Result<()> {
      // ...
  }

  // ❌ Bad - unnecessary clone
  fn process_task(task: Task) -> Result<()> {
      // ...
  }
  ```

- **Use `Arc` for shared ownership, not `Rc` in async code**
- **Use `tokio::spawn_blocking` for CPU-intensive work**
  ```rust
  let content_hash = tokio::task::spawn_blocking(move || {
      sha256(&large_content)
  }).await?;
  ```

- **Batch Redis operations where possible**
  ```rust
  // ✅ Good - pipeline
  let mut pipe = redis::pipe();
  pipe.set("key1", "value1")
      .set("key2", "value2")
      .set("key3", "value3");
  pipe.query_async(&mut conn).await?;

  // ❌ Bad - multiple round trips
  conn.set("key1", "value1").await?;
  conn.set("key2", "value2").await?;
  conn.set("key3", "value3").await?;
  ```

### Security (C-SECURITY)

- **Validate all external input**
  ```rust
  use url::Url;

  pub fn validate_url(url: &str) -> Result<Url, FetchError> {
      let parsed = Url::parse(url)
          .map_err(|_| FetchError::InvalidUrl(url.to_string()))?;

      // Only allow HTTP(S)
      if parsed.scheme() != "http" && parsed.scheme() != "https" {
          return Err(FetchError::InvalidUrl("Only HTTP(S) allowed".into()));
      }

      Ok(parsed)
  }
  ```

- **Never log sensitive data** (passwords, tokens, API keys)
  ```rust
  // ✅ Good
  #[derive(Debug)]
  pub struct ProxyConfig {
      pub url: String,
      #[debug(skip)]  // Don't include in Debug output
      pub password: String,
  }

  // ❌ Bad
  info!("Using proxy credentials: {}:{}", user, password);
  ```

- **Use `secrecy` crate for sensitive data**
  ```rust
  use secrecy::{Secret, ExposeSecret};

  pub struct ProxyAuth {
      username: String,
      password: Secret<String>,
  }

  impl ProxyAuth {
      pub fn authenticate(&self, client: &mut Client) {
          client.basic_auth(&self.username, Some(self.password.expose_secret()));
      }
  }
  ```

### Code Organization (C-ORGANIZATION)

- **One module per file in `src/` directory**
- **Use `mod.rs` for module public interface**
  ```
  fetcher/
  ├── mod.rs           # pub use static_fetch::*, browser_fetch::*
  ├── static_fetch.rs  # HTTP fetching implementation
  ├── browser_fetch.rs # Browser rendering implementation
  └── adaptive.rs      # Fallback classifier
  ```

- **Keep functions focused and small** (<100 lines)
- **Extract complex logic into helper functions**
- **Use meaningful variable names**
  ```rust
  // ✅ Good
  let content_hash = compute_sha256(&response.body);
  let cache_key = format!("cache:content_hash:{}:{}", domain, url_hash);

  // ❌ Bad
  let h = compute_sha256(&resp.b);
  let k = format!("cache:content_hash:{}:{}", d, u);
  ```

---

## Python Standards

### Type Hints (PY-TYPES)

- **All function signatures must have type hints**
  ```python
  # ✅ Good
  async def fetch_task(task_id: str, redis: Redis) -> Task | None:
      data = await redis.get(f"task:{task_id}")
      if data is None:
          return None
      return Task.parse_raw(data)

  # ❌ Bad
  async def fetch_task(task_id, redis):
      data = await redis.get(f"task:{task_id}")
      if data is None:
          return None
      return Task.parse_raw(data)
  ```

- **Use `|` for union types (Python 3.10+)**
  ```python
  def process_result(result: TaskResult | None) -> bool:
      return result is not None
  ```

- **Use generics where appropriate**
  ```python
  from typing import TypeVar, Generic

  T = TypeVar('T')

  class CacheEntry(Generic[T]):
      value: T
      expires_at: datetime
  ```

### Async/Await (PY-ASYNC)

- **Use async/await throughout** (FastAPI, asyncpg, redis.asyncio)
  ```python
  # ✅ Good
  async def create_campaign(campaign: Campaign, db: asyncpg.Pool) -> UUID:
      async with db.acquire() as conn:
          row = await conn.fetchrow(
              "INSERT INTO campaigns (name, schedule) VALUES ($1, $2) RETURNING id",
              campaign.name, campaign.schedule
          )
          return row['id']

  # ❌ Bad - mixing sync and async
  def create_campaign(campaign: Campaign, db: Pool) -> UUID:
      conn = db.get_connection()
      row = conn.execute("INSERT...")  # Blocking!
      return row['id']
  ```

- **Use `asyncio.gather()` for concurrent operations**
  ```python
  results = await asyncio.gather(
      fetch_from_redis(task_id),
      fetch_from_s3(s3_key),
      fetch_metadata(campaign_id),
      return_exceptions=True
  )
  ```

### Pydantic Models (PY-MODELS)

- **Use Pydantic for all data models**
  ```python
  from pydantic import BaseModel, Field, validator
  from datetime import datetime
  from uuid import UUID

  class Campaign(BaseModel):
      id: UUID | None = None
      name: str = Field(..., min_length=1, max_length=255)
      schedule: str
      priority: int = Field(default=50, ge=0, le=100)
      enabled: bool = True
      created_at: datetime = Field(default_factory=datetime.now)

      @validator('schedule')
      def validate_schedule(cls, v):
          if not (v.startswith('cron:') or v.startswith('continuous:')):
              raise ValueError('Invalid schedule format')
          return v

      class Config:
          orm_mode = True
  ```

- **Use Pydantic Settings for configuration**
  ```python
  from pydantic_settings import BaseSettings

  class Settings(BaseSettings):
      redis_url: str = "redis://localhost:6379"
      database_url: str
      s3_bucket: str
      s3_endpoint: str | None = None

      class Config:
          env_file = ".env"
          env_file_encoding = "utf-8"
  ```

### Error Handling (PY-ERRORS)

- **Never use bare `except:`**
  ```python
  # ✅ Good
  try:
      result = await redis.get(key)
  except redis.RedisError as e:
      logger.error("Redis error", error=str(e), key=key)
      raise
  except Exception as e:
      logger.exception("Unexpected error", key=key)
      raise

  # ❌ Bad
  try:
      result = await redis.get(key)
  except:
      pass
  ```

- **Use custom exceptions for domain errors**
  ```python
  class ArachneError(Exception):
      """Base exception for Arachne"""
      pass

  class RateLimitExceeded(ArachneError):
      def __init__(self, domain: str, retry_after: int):
          self.domain = domain
          self.retry_after = retry_after
          super().__init__(f"Rate limit exceeded for {domain}, retry after {retry_after}s")
  ```

### Logging (PY-LOGGING)

- **Use `structlog` for all logging**
  ```python
  import structlog

  logger = structlog.get_logger(__name__)

  async def process_task(task: Task) -> None:
      log = logger.bind(task_id=str(task.id), domain=task.domain)
      log.info("processing_task_started")

      try:
          result = await fetch_url(task.url)
          log.info("task_completed", status_code=result.status)
      except Exception as e:
          log.error("task_failed", error=str(e), exc_info=True)
          raise
  ```

- **Never use `print()` in production code**
- **Use structured logging with key-value pairs**

### Code Organization (PY-ORGANIZATION)

- **One class per file for large classes**
- **Keep modules focused** (< 500 lines)
- **Use `__init__.py` for public API exports**
  ```python
  # arachne/extraction/__init__.py
  from .pipeline import ExtractionPipeline
  from .worker import ExtractionWorker
  from .dsl import ExtractionDSL

  __all__ = [
      "ExtractionPipeline",
      "ExtractionWorker",
      "ExtractionDSL",
  ]
  ```

### Testing (PY-TESTING)

- **Use `pytest` with `pytest-asyncio`**
  ```python
  import pytest
  from arachne.api.routes import campaigns

  @pytest.mark.asyncio
  async def test_create_campaign_success(db_pool, test_campaign):
      campaign_id = await campaigns.create_campaign(test_campaign, db_pool)
      assert campaign_id is not None

      # Verify in database
      async with db_pool.acquire() as conn:
          row = await conn.fetchrow("SELECT * FROM campaigns WHERE id = $1", campaign_id)
          assert row['name'] == test_campaign.name

  @pytest.mark.asyncio
  async def test_create_campaign_invalid_schedule(db_pool):
      invalid_campaign = Campaign(name="Test", schedule="invalid")
      with pytest.raises(ValueError):
          await campaigns.create_campaign(invalid_campaign, db_pool)
  ```

- **Use fixtures for common setup**
  ```python
  # conftest.py
  import pytest
  import fakeredis.aioredis

  @pytest.fixture
  async def redis_client():
      client = fakeredis.aioredis.FakeRedis()
      yield client
      await client.close()

  @pytest.fixture
  def test_campaign():
      return Campaign(
          name="Test Campaign",
          schedule="cron:0 0 * * *",
          domain_list=["example.com"]
      )
  ```

---

## General Principles

### Code Review Checklist

Before submitting code for review, verify:

- [ ] All public APIs are documented
- [ ] Error handling uses `Result` (Rust) or explicit exception handling (Python)
- [ ] No `unwrap()`, `expect()`, or bare `except:` in production code
- [ ] All async functions use appropriate runtime (tokio for Rust, asyncio for Python)
- [ ] Logging uses structured fields, not string interpolation
- [ ] Tests pass locally: `cargo test --workspace` (Rust), `pytest` (Python)
- [ ] Linting passes: `cargo clippy` (Rust), `ruff check` (Python)
- [ ] No sensitive data in logs or error messages
- [ ] Resource cleanup is automatic (RAII, context managers, Drop traits)
- [ ] Performance considerations: avoid unnecessary clones, use batching for external calls

### Security Checklist

- [ ] All external input is validated
- [ ] No SQL injection vulnerabilities (use parameterized queries)
- [ ] No command injection vulnerabilities (avoid shell=True, use subprocess safely)
- [ ] Secrets are never logged or included in error messages
- [ ] TLS certificate validation is enabled (no `danger_accept_invalid_certs`)
- [ ] Rate limiting is enforced at application layer
- [ ] CORS policies are restrictive (if applicable)
- [ ] Authentication is required for sensitive endpoints

### Performance Checklist

- [ ] Database queries use indexes on filter columns
- [ ] Redis operations are batched where possible (pipelines)
- [ ] Large payloads are streamed, not loaded into memory
- [ ] HTTP connection pooling is used (reqwest Client reuse)
- [ ] Browser instances are recycled after N page loads
- [ ] Metrics are recorded for all critical paths
- [ ] Backpressure mechanisms prevent queue overflow

---

## Common Pitfalls to Avoid

### Rust

❌ **Using `unwrap()` in production**
```rust
let value = map.get("key").unwrap();  // Will panic if key missing!
```
✅ **Use proper error handling**
```rust
let value = map.get("key").ok_or(Error::KeyNotFound)?;
```

❌ **Blocking in async context**
```rust
async fn process() {
    std::thread::sleep(Duration::from_secs(1));  // Blocks executor!
}
```
✅ **Use async sleep**
```rust
async fn process() {
    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

❌ **Mixing `String` and `&str` unnecessarily**
```rust
fn process(s: &String) { }  // Unnecessary restriction
```
✅ **Accept `&str` for flexibility**
```rust
fn process(s: &str) { }  // Can accept &String, &str, String
```

### Python

❌ **Mixing sync and async**
```python
async def fetch():
    result = requests.get(url)  # Blocking call in async function!
```
✅ **Use async client**
```python
async def fetch():
    async with httpx.AsyncClient() as client:
        result = await client.get(url)
```

❌ **Mutating during iteration**
```python
for task in tasks:
    if task.failed:
        tasks.remove(task)  # Modifies list during iteration!
```
✅ **Filter or iterate over copy**
```python
tasks = [t for t in tasks if not t.failed]
```

❌ **Not using context managers for resources**
```python
conn = db.connect()
conn.execute(query)
conn.close()  # May not run if exception occurs!
```
✅ **Use context manager**
```python
async with db.acquire() as conn:
    await conn.execute(query)
```

---

## Enforcement

- **Pre-commit hooks** run `cargo clippy`, `ruff check`, formatters
- **CI pipeline** blocks merge if:
  - Tests fail
  - Linting fails
  - Code coverage drops below 80%
  - Security scan finds critical issues
- **Code review** enforces these standards before approval

---

## Questions?

Refer to:
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Microsoft Rust Guidelines](https://microsoft.github.io/rust-guidelines/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

For project-specific questions, consult `CLAUDE.md` or the team lead.
