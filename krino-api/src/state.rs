use crate::auth::ApiKeyStore;
use crate::config::ApiConfig;
use crate::metrics::MetricsState;
use crate::worker_pool::WorkerPool;
use krino::modules::groundedness::GroundednessConfig;
use std::sync::Arc;

/// Shared application state, initialized once at startup.
/// Cloned (Arc) into every request handler.
#[derive(Clone)]
pub struct AppState {
    /// Worker pool — owns all ONNX sessions and Rayon thread pools.
    pub worker_pool: Arc<WorkerPool>,

    /// Server configuration
    pub config: ApiConfig,

    /// API key store (in-memory for v1, DynamoDB later)
    pub auth: Arc<ApiKeyStore>,

    /// Request metrics
    pub metrics: Arc<MetricsState>,
}

impl AppState {
    pub fn load(config: ApiConfig) -> anyhow::Result<Self> {
        let nli_path = std::path::Path::new(&config.models.nli_model_path);
        let embedding_path = std::path::Path::new(&config.models.embedding_model_path);

        log_cpu_diagnostics();

        tracing::info!(
            n_workers = config.workers.n_workers,
            threads_per_worker = config.workers.threads_per_worker,
            queue_depth = config.workers.queue_depth,
            "Starting worker pool"
        );

        let faithfulness_config = GroundednessConfig {
            top_k_context: config.faithfulness.default_top_k,
            contradiction_threshold: config.faithfulness.contradiction_threshold,
            treat_neutral_as_unsupported: false,
            min_claim_length: 10,
            min_similarity_threshold: config.faithfulness.min_similarity_threshold,
            adaptive_top_k: config.faithfulness.adaptive_top_k,
            include_entailment_matrix: false,
            flag_compound_claims: true,
            partial_threshold: config.faithfulness.partial_threshold,
            partial_neutral_ceiling: config.faithfulness.partial_neutral_ceiling,
            partial_similarity_floor: config.faithfulness.partial_similarity_floor,
        };

        let worker_pool = Arc::new(WorkerPool::new(
            &config.workers,
            nli_path,
            embedding_path,
            faithfulness_config,
        )?);

        tracing::info!(n_workers = config.workers.n_workers, "Worker pool ready");

        let auth = Arc::new(ApiKeyStore::from_config(&config)?);
        let metrics = Arc::new(MetricsState::new()?);

        tracing::info!("Krino API ready");

        Ok(Self {
            worker_pool,
            config,
            auth,
            metrics,
        })
    }
}

fn log_cpu_diagnostics() {
    #[cfg(target_arch = "x86_64")]
    {
        tracing::info!(
            avx2 = is_x86_feature_detected!("avx2"),
            avx512f = is_x86_feature_detected!("avx512f"),
            avx512vnni = is_x86_feature_detected!("avx512vnni"),
            fma = is_x86_feature_detected!("fma"),
            cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            "CPU capabilities"
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        tracing::info!(
            arch = std::env::consts::ARCH,
            cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            "CPU capabilities"
        );
    }
}
