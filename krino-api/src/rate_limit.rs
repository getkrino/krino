use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::auth::CustomerId;
use crate::error::ApiError;
use crate::state::AppState;

/// Token bucket rate limiter
pub struct RateLimiter {
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
    requests_per_minute: u64,
    burst: u64,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u64, burst: u64) -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            requests_per_minute,
            burst,
        }
    }

    /// Check if a request is allowed for the given customer ID
    pub async fn check_rate_limit(&self, customer_id: &str) -> Result<(), u64> {
        let mut buckets = self.buckets.lock().await;
        let bucket = buckets
            .entry(customer_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.burst, self.requests_per_minute));

        if bucket.try_consume() {
            Ok(())
        } else {
            // Calculate retry_after based on refill rate
            let retry_after_secs = 60 / self.requests_per_minute;
            Err(retry_after_secs)
        }
    }
}

/// Token bucket for rate limiting
struct TokenBucket {
    tokens: f64,
    capacity: u64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u64, requests_per_minute: u64) -> Self {
        let refill_rate = requests_per_minute as f64 / 60.0;
        Self {
            tokens: capacity as f64,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f64);
        self.last_refill = now;
    }

    fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // Extract customer ID from request extensions (set by auth middleware)
    let customer_id = req
        .extensions()
        .get::<CustomerId>()
        .map(|id| id.0.clone())
        .unwrap_or_else(|| "anonymous".to_string());

    // Create rate limiter on-the-fly (in production, would be in AppState)
    let rate_limiter = RateLimiter::new(
        state.config.rate_limit.requests_per_minute,
        state.config.rate_limit.burst,
    );

    match rate_limiter.check_rate_limit(&customer_id).await {
        Ok(()) => Ok(next.run(req).await),
        Err(retry_after_secs) => Err(ApiError::rate_limited(retry_after_secs)),
    }
}
