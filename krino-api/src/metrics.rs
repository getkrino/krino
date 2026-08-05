use anyhow::Context as _;
use metrics::{counter, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use std::time::Instant;

pub struct MetricsState {
    handle: PrometheusHandle,
}

impl MetricsState {
    pub fn new() -> anyhow::Result<Self> {
        let handle = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full("http_request_duration_seconds".to_string()),
                &[
                    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                ],
            )
            .context("failed to set buckets for http_request_duration_seconds")?
            .set_buckets_for_metric(
                Matcher::Full("nli_inference_latency_seconds".to_string()),
                &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            )
            .context("failed to set buckets for nli_inference_latency_seconds")?
            .install_recorder()
            .context("failed to install Prometheus recorder")?;

        Ok(Self { handle })
    }

    pub fn render(&self) -> String {
        self.handle.render()
    }
}

/// Records an HTTP request with method, path, and status
pub fn record_request(method: &str, path: &str, status: u16, duration: std::time::Duration) {
    counter!(
        "http_requests_total",
        "method" => method.to_string(),
        "path" => path.to_string(),
        "status" => status.to_string(),
    )
    .increment(1);

    histogram!(
        "http_request_duration_seconds",
        "method" => method.to_string(),
        "path" => path.to_string(),
    )
    .record(duration.as_secs_f64());
}

/// Records total NLI inference latency for one evaluation (all batches combined)
pub fn record_nli_inference(latency_ms: f64) {
    histogram!("nli_inference_latency_seconds").record(latency_ms / 1000.0);
}

/// Records an evaluation request
pub fn record_evaluation(eval_type: &str, duration: std::time::Duration, claims: usize) {
    counter!(
        "evaluations_total",
        "type" => eval_type.to_string(),
    )
    .increment(1);

    histogram!(
        "evaluation_duration_seconds",
        "type" => eval_type.to_string(),
    )
    .record(duration.as_secs_f64());

    histogram!(
        "evaluation_claims",
        "type" => eval_type.to_string(),
    )
    .record(claims as f64);
}

/// Timer helper for measuring durations
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
