use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    #[serde(default)]
    pub server: ServerConfig,

    pub models: ModelsConfig,

    #[serde(default)]
    pub faithfulness: FaithfulnessApiConfig,

    #[serde(default)]
    pub workers: WorkerPoolConfig,

    pub auth: AuthConfig,

    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    pub nli_model_path: String,

    /// Path to sentence embedding model (all-MiniLM-L6-v2) for pre-filtering.
    /// Required — pre-filtering must be enabled for large context inputs.
    pub embedding_model_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FaithfulnessApiConfig {
    #[serde(default = "default_top_k")]
    pub default_top_k: usize,

    #[serde(default = "default_threshold")]
    pub default_threshold: f64,

    /// Minimum contradiction probability for a claim to be marked as
    /// contradicted (vs neutral) in the engine's per-claim verdict. Tuned
    /// per NLI model — FP32 DeBERTa-MNLI hits 0.95+ on contradictions, but
    /// INT8 quantization softens the confidence band to ~0.55-0.6, so the
    /// historical 0.7 default mis-classifies real contradictions as neutral.
    /// 0.5 catches INT8 quantized models while still keeping FP32 behavior
    /// (which clears 0.5 by a wide margin).
    #[serde(default = "default_contradiction_threshold")]
    pub contradiction_threshold: f64,

    /// Minimum cosine similarity for a context sentence to be considered as
    /// an NLI candidate. Raising this cuts NLI calls; setting too high drops
    /// borderline-relevant context. 0.25 is the production-tuned default.
    #[serde(default = "default_min_similarity_threshold")]
    pub min_similarity_threshold: f32,

    /// When true, the per-claim top-K is scaled by claim length:
    /// `clamp(ceil(claim_chars / 40), 3, default_top_k)`. Short claims get
    /// fewer candidates, which cuts NLI calls without changing verdicts on
    /// the long tail.
    #[serde(default = "default_adaptive_top_k")]
    pub adaptive_top_k: bool,

    /// Floor on per-pair entailment for partial-verdict eligibility. Acts as
    /// a guardrail under the three-condition partial rule; the rule itself
    /// (entailment > contradiction AND neutral < ceiling AND similarity >=
    /// floor) does the bulk of the gating. See engine docs.
    #[serde(default = "default_partial_threshold")]
    pub partial_threshold: f64,

    /// Upper bound on per-pair neutral probability for partial-verdict
    /// eligibility. See engine docs for empirical calibration.
    #[serde(default = "default_partial_neutral_ceiling")]
    pub partial_neutral_ceiling: f64,

    /// Lower bound on embedding similarity for partial-verdict eligibility.
    /// No effect when pre-filtering is disabled.
    #[serde(default = "default_partial_similarity_floor")]
    pub partial_similarity_floor: f32,

    #[serde(default = "default_max_context")]
    pub max_context_chars: usize,

    #[serde(default = "default_max_output")]
    pub max_output_chars: usize,

    /// Available granularities: ["claim"]. Add "token" when ModernBERT model ships.
    #[serde(default = "default_available_granularities")]
    pub available_granularities: Vec<String>,
}

impl Default for FaithfulnessApiConfig {
    fn default() -> Self {
        Self {
            default_top_k: default_top_k(),
            default_threshold: default_threshold(),
            contradiction_threshold: default_contradiction_threshold(),
            min_similarity_threshold: default_min_similarity_threshold(),
            adaptive_top_k: default_adaptive_top_k(),
            partial_threshold: default_partial_threshold(),
            partial_neutral_ceiling: default_partial_neutral_ceiling(),
            partial_similarity_floor: default_partial_similarity_floor(),
            max_context_chars: default_max_context(),
            max_output_chars: default_max_output(),
            available_granularities: default_available_granularities(),
        }
    }
}

fn default_available_granularities() -> Vec<String> {
    vec!["claim".to_string()]
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    pub api_keys: Vec<ApiKeyEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeyEntry {
    pub customer_id: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default = "default_requests_per_minute")]
    pub requests_per_minute: u64,

    #[serde(default = "default_burst")]
    pub burst: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: default_requests_per_minute(),
            burst: default_burst(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,

    #[serde(default = "default_log_format")]
    pub format: String,

    #[serde(default = "default_ort_level")]
    pub ort_level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            ort_level: default_ort_level(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerPoolConfig {
    /// Number of independent inference workers.
    /// Each worker owns its own ONNX session and Rayon thread pool.
    /// Default: vCPUs / 4, minimum 1.
    #[serde(default = "default_n_workers")]
    pub n_workers: usize,

    /// Rayon threads per worker.
    /// Total thread budget = n_workers * threads_per_worker.
    /// Default: vCPUs / n_workers.
    #[serde(default = "default_threads_per_worker")]
    pub threads_per_worker: usize,

    /// Bounded channel depth. When full, requests receive 503 immediately.
    /// Default: n_workers * 2.
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,

    /// NLI pairs per ORT forward pass. Larger batches amortize ORT overhead.
    /// Default: 8.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            n_workers: default_n_workers(),
            threads_per_worker: default_threads_per_worker(),
            queue_depth: default_queue_depth(),
            batch_size: default_batch_size(),
        }
    }
}

/// One worker, sized to maximize single-request latency.
///
/// The tradeoff between worker count and threads-per-worker is single-request
/// latency vs. concurrent throughput. For the demo box (typical load: one
/// benchmark or one user at a time) the right pick is one worker that takes
/// every vCPU it can. A second concurrent request will queue, by design.
///
/// To prioritize throughput instead, override `[workers].n_workers` and
/// `[workers].threads_per_worker` in `krino-api.toml`.
fn default_n_workers() -> usize {
    1
}

/// All available vCPUs minus one reserved for tokio/OS overhead.
///
/// `available_parallelism` already accounts for cgroup-pinned cores in
/// container environments, so this is correct under both bare-metal and
/// Docker on the demo box.
fn default_threads_per_worker() -> usize {
    let vcpus = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    vcpus.saturating_sub(1).max(1)
}

fn default_queue_depth() -> usize {
    default_n_workers() * 2
}

fn default_batch_size() -> usize {
    16
}

// Default functions
fn default_port() -> u16 {
    8080
}

fn default_top_k() -> usize {
    10
}

fn default_threshold() -> f64 {
    0.7
}

fn default_contradiction_threshold() -> f64 {
    0.5
}

fn default_min_similarity_threshold() -> f32 {
    0.25
}

fn default_adaptive_top_k() -> bool {
    true
}

fn default_partial_threshold() -> f64 {
    0.2
}

fn default_partial_neutral_ceiling() -> f64 {
    0.65
}

fn default_partial_similarity_floor() -> f32 {
    0.7
}

fn default_max_context() -> usize {
    500_000
}

fn default_max_output() -> usize {
    50_000
}

fn default_requests_per_minute() -> u64 {
    60
}

fn default_burst() -> u64 {
    10
}

fn default_log_level() -> String {
    "info".into()
}

fn default_log_format() -> String {
    "json".into()
}

fn default_ort_level() -> String {
    "warn".into()
}
