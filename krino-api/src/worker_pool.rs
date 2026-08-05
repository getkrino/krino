use std::path::Path;
use std::sync::Arc;

use krino::models::backends::onnx::{OnnxConfig, OnnxEmbeddingBackend, OnnxSequenceClassifier};
use krino::models::inference::EmbeddingSimilarity;
use krino::modules::groundedness::{
    GroundednessChecker, GroundednessConfig, GroundednessResult, RequestOverrides,
};

use crate::config::WorkerPoolConfig;
use crate::error::ApiError;

struct WorkItem {
    context: String,
    output: String,
    overrides: RequestOverrides,
    reply: tokio::sync::oneshot::Sender<Result<GroundednessResult, String>>,
}

pub struct WorkerPool {
    tx: async_channel::Sender<WorkItem>,
}

impl WorkerPool {
    pub fn new(
        config: &WorkerPoolConfig,
        nli_model_path: &Path,
        embedding_model_path: &Path,
        faithfulness_config: GroundednessConfig,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = async_channel::bounded::<WorkItem>(config.queue_depth);

        for worker_idx in 0..config.n_workers {
            let rx = rx.clone();
            let faithfulness_config = faithfulness_config.clone();
            let threads = config.threads_per_worker;

            // Each worker loads its own models — no sharing between workers.
            let onnx_config = OnnxConfig {
                intra_op_num_threads: threads,
                batch_size: config.batch_size,
            };

            let nli_backend = Arc::new(
                OnnxSequenceClassifier::from_pretrained_quantized_with_config(
                    nli_model_path,
                    onnx_config,
                )?,
            );

            let embedding_backend = Arc::new(OnnxEmbeddingBackend::from_pretrained_quantized(
                embedding_model_path,
            )?) as Arc<dyn EmbeddingSimilarity>;

            let checker =
                GroundednessChecker::new(nli_backend, embedding_backend, faithfulness_config);

            // Isolated Rayon pool — never touches the global pool.
            let rayon_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |i: usize| format!("krino-worker{worker_idx}-rayon{i}"))
                .build()
                .map_err(|e| {
                    anyhow::anyhow!("Failed to build Rayon pool for worker {worker_idx}: {e}")
                })?;

            std::thread::Builder::new()
                .name(format!("krino-worker{worker_idx}"))
                .spawn(move || {
                    while let Ok(item) = rx.recv_blocking() {
                        let result = rayon_pool
                            .install(|| {
                                checker.check_with_overrides(
                                    &item.context,
                                    &item.output,
                                    item.overrides,
                                )
                            })
                            .map_err(|e| e.to_string());
                        // Receiver gone means the request timed out or was dropped — ignore.
                        let _ = item.reply.send(result);
                    }
                    tracing::info!(worker = worker_idx, "Worker shut down");
                })?;

            tracing::info!(worker = worker_idx, threads, "Worker started");
        }

        Ok(Self { tx })
    }

    pub async fn evaluate(
        &self,
        context: String,
        output: String,
        overrides: RequestOverrides,
    ) -> Result<GroundednessResult, ApiError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();

        self.tx
            .try_send(WorkItem {
                context,
                output,
                overrides,
                reply: reply_tx,
            })
            .map_err(|_| {
                ApiError::service_unavailable("All inference workers are busy. Retry in a moment.")
            })?;

        reply_rx
            .await
            .map_err(|_| ApiError::internal("Worker died before returning a result"))?
            .map_err(ApiError::internal)
    }

    pub fn queue_depth(&self) -> usize {
        self.tx.len()
    }

    pub fn is_full(&self) -> bool {
        self.tx.is_full()
    }
}
