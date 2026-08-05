//! ONNX Runtime inference backend.
//!
//! This module implements `SequenceClassifier` using ONNX Runtime via the 'ort' crate.
//! Primary use case: DeBERTa-v3-large for NLI-based groundedness checking.

use crate::error::{ModelError, Result};
use crate::models::inference::{
    EmbeddingSimilarity, SequenceClassifier, SequenceClassifierInput, SequenceClassifierOutput,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Computes the engine-usable max sequence length from the model config.
///
/// `RoBERTa` (and XLM-R) reserve position indices `0..=pad_token_id` for padding,
/// so their usable cap is `max_position_embeddings - pad_token_id - 1`.
/// `RoBERTa`-large reports `max_position_embeddings = 514` but the position
/// embedding table only addresses 0..513 — feeding the model a 514-token
/// sequence causes a Gather out-of-bounds at runtime.
///
/// Other architectures (BERT, `DeBERTa`, etc.) use straight `0..max_position`
/// indexing with no offset.
#[allow(clippy::cast_possible_truncation)]
fn effective_max_length(config: &serde_json::Value) -> usize {
    let max_pos = config["max_position_embeddings"].as_u64().unwrap_or(1024) as usize;
    let offset = match config["model_type"].as_str().unwrap_or("") {
        "roberta" | "xlm-roberta" => (config["pad_token_id"].as_u64().unwrap_or(1) as usize) + 1,
        _ => 0,
    };
    max_pos.saturating_sub(offset)
}

/// Numerically stable softmax over f32 logits.
///
/// Uses the max-subtraction trick to prevent overflow.
/// Returns f64 probabilities for downstream precision.
///
/// # Arguments
///
/// * `logits` - Raw logits from the model
///
/// # Returns
///
/// Probability distribution (sums to 1.0)
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits
        .iter()
        .map(|&x| f64::from((x - max_logit).exp() / exp_sum))
        .collect()
}

/// Configuration for ONNX inference performance tuning.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Total intra-op threads budget, split evenly across the session pool.
    ///
    /// Defaults to the number of logical CPUs reported by the OS.
    /// Increase on instances with more vCPUs; decrease on memory-constrained hosts.
    pub intra_op_num_threads: usize,

    /// Batch size for inference.
    /// Default: 8
    /// Larger batches improve throughput but increase latency.
    /// Optimal range for DeBERTa-v3-large on CPU: 8-16
    pub batch_size: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        let vcpus = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        Self {
            intra_op_num_threads: vcpus,
            batch_size: 8,
        }
    }
}

pub struct OnnxSequenceClassifier {
    /// Pool of `NUM_SESSIONS` independent ORT sessions.
    ///
    /// Each session runs on a dedicated Rayon worker thread with
    /// `intra_op_num_threads / NUM_SESSIONS` intra-op threads, so total
    /// thread usage stays bounded while inference runs in parallel.
    sessions: Vec<Arc<Mutex<Session>>>,
    tokenizer: Tokenizer,
    max_length: usize,
    label_names: Vec<String>,
    has_token_type_ids: bool,
    config: OnnxConfig,
    /// Permutation that reorders the model's raw logit indices into the
    /// engine's canonical `[entailment, neutral, contradiction]` order.
    /// `label_perm[i]` is the index in the model's output that corresponds
    /// to canonical class `i`.
    ///
    /// DeBERTa-MNLI ships id2label = `{0: entailment, 1: neutral, 2: contradiction}`
    /// → perm = `[0, 1, 2]` (identity).
    /// RoBERTa-MNLI ships id2label = `{0: contradiction, 1: neutral, 2: entailment}`
    /// → perm = `[2, 1, 0]`.
    ///
    /// Built from id2label *names* at load time so the engine works with any
    /// MNLI model regardless of label ordering convention. Downstream
    /// `groundedness.rs` uses constants `LABEL_ENTAILMENT=0, LABEL_NEUTRAL=1,
    /// LABEL_CONTRADICTION=2` and assumes probs are already in this order —
    /// `classify()` applies the permutation before returning.
    label_perm: [usize; 3],
}

/// Run a single ORT forward pass on a pre-padded flat buffer slice.
///
/// Returns one `Vec<f32>` of logits per input row in the chunk.
fn run_ort_batch_slice(
    session: &Arc<Mutex<Session>>,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
    chunk_size: usize,
    max_seq_len: usize,
    has_token_type_ids: bool,
) -> Result<Vec<Vec<f32>>> {
    let input_ids_tensor =
        ort::value::Value::from_array(([chunk_size, max_seq_len], input_ids.to_vec()))
            .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

    let attention_mask_tensor =
        ort::value::Value::from_array(([chunk_size, max_seq_len], attention_mask.to_vec()))
            .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

    let ort_inputs = if has_token_type_ids {
        let token_type_ids_tensor =
            ort::value::Value::from_array(([chunk_size, max_seq_len], token_type_ids.to_vec()))
                .map_err(|e| ModelError::inference_failed(format!("{e}")))?;
        ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ]
    } else {
        ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ]
    };

    let mut session_guard = session
        .lock()
        .map_err(|e| ModelError::inference_failed(format!("Mutex lock failed: {e}")))?;

    let ort_outputs = session_guard
        .run(ort_inputs)
        .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

    let logits_tensor = ort_outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

    let logits_shape = logits_tensor.0;
    let logits_data = &logits_tensor.1;

    let num_labels = if logits_shape.len() >= 2 {
        usize::try_from(logits_shape[1]).unwrap_or(logits_data.len() / chunk_size)
    } else {
        logits_data.len() / chunk_size
    };

    let mut all_logits = Vec::with_capacity(chunk_size);
    for i in 0..chunk_size {
        let start = i * num_labels;
        let end = start + num_labels;
        if end <= logits_data.len() {
            all_logits.push(logits_data[start..end].to_vec());
        } else {
            let mut logits = logits_data[start..logits_data.len()].to_vec();
            while logits.len() < num_labels {
                logits.push(0.0);
            }
            all_logits.push(logits);
        }
    }

    Ok(all_logits)
}

impl OnnxSequenceClassifier {
    /// Loads an ONNX sequence classifier from a directory.
    ///
    /// # Expected directory structure
    ///
    /// ```text
    /// model_path/
    ///   ├── model.onnx
    ///   ├── tokenizer.json
    ///   └── config.json
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model files are missing
    /// - ONNX model is invalid or incompatible
    /// - Config is malformed
    pub fn from_pretrained(model_path: &Path) -> Result<Self> {
        Self::from_pretrained_with_config(model_path, OnnxConfig::default())
    }

    /// Loads a quantized ONNX sequence classifier (`model_quantized.onnx`).
    pub fn from_pretrained_quantized(model_path: &Path) -> Result<Self> {
        Self::from_pretrained_quantized_with_config(model_path, OnnxConfig::default())
    }

    /// Loads a quantized ONNX sequence classifier with custom configuration.
    pub fn from_pretrained_quantized_with_config(
        model_path: &Path,
        onnx_config: OnnxConfig,
    ) -> Result<Self> {
        Self::load(model_path, "model_quantized.onnx", onnx_config)
    }

    /// Loads an ONNX sequence classifier with custom configuration.
    ///
    /// See `from_pretrained` for directory structure requirements.
    pub fn from_pretrained_with_config(model_path: &Path, onnx_config: OnnxConfig) -> Result<Self> {
        Self::load(model_path, "model.onnx", onnx_config)
    }

    fn load(model_path: &Path, model_filename: &str, onnx_config: OnnxConfig) -> Result<Self> {
        let intra_threads = onnx_config.intra_op_num_threads.max(1);

        info!(
            "Loading ONNX backend from: {} (intra_threads={}, batch_size={})",
            model_path.display(),
            intra_threads,
            onnx_config.batch_size
        );

        let model_file = model_path.join(model_filename);

        let session = Session::builder()
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .with_intra_threads(intra_threads)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .commit_from_file(&model_file)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?;

        let sessions = vec![Arc::new(Mutex::new(session))];

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| ModelError::load_failed(&tokenizer_path, e.to_string()))?;

        // Load config to get label mapping and max_length
        let config_path = model_path.join("config.json");
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?,
        )?;

        // Build the label permutation from id2label *names*. Different MNLI
        // models ship the same three classes in different orders (DeBERTa
        // entailment-first, RoBERTa contradiction-first); the engine treats
        // index 0 as entailment, 1 as neutral, 2 as contradiction, so we
        // permute the model's raw logits to match.
        let id2label = config["id2label"]
            .as_object()
            .ok_or_else(|| ModelError::load_failed(&config_path, "Missing id2label"))?;

        let mut raw_labels: Vec<(usize, String)> = id2label
            .iter()
            .map(|(k, v)| {
                let idx = k.parse::<usize>().map_err(|_| {
                    ModelError::load_failed(&config_path, format!("Invalid label index: {k}"))
                })?;
                let name = v
                    .as_str()
                    .ok_or_else(|| {
                        ModelError::load_failed(
                            &config_path,
                            format!("Invalid label name for index {k}"),
                        )
                    })?
                    .to_string();
                Ok((idx, name))
            })
            .collect::<Result<Vec<_>>>()?;
        raw_labels.sort_by_key(|(idx, _)| *idx);

        let find_label = |needle: &str| -> Result<usize> {
            raw_labels
                .iter()
                .find(|(_, name)| name.eq_ignore_ascii_case(needle))
                .map(|(idx, _)| *idx)
                .ok_or_else(|| {
                    ModelError::load_failed(
                        &config_path,
                        format!(
                            "id2label is missing '{needle}'. Got {raw_labels:?}. \
                             This backend only supports 3-class MNLI models \
                             with labels entailment / neutral / contradiction."
                        ),
                    )
                    .into()
                })
        };
        let label_perm: [usize; 3] = [
            find_label("entailment")?,
            find_label("neutral")?,
            find_label("contradiction")?,
        ];
        let label_names: Vec<String> = vec![
            "entailment".to_string(),
            "neutral".to_string(),
            "contradiction".to_string(),
        ];

        let max_length = effective_max_length(&config);

        // Introspect ONNX model inputs to check if token_type_ids is required.
        // All sessions share the same model, so inspecting the first is sufficient.
        let has_token_type_ids = sessions[0]
            .lock()
            .map_err(|e| ModelError::load_failed(&model_file, format!("Mutex lock failed: {e}")))?
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        info!(
            "ONNX backend loaded: 3 labels ({:?}), max_length={}, token_type_ids={}, label_perm={:?}",
            label_names, max_length, has_token_type_ids, label_perm,
        );

        Ok(Self {
            sessions,
            tokenizer,
            max_length,
            label_names,
            has_token_type_ids,
            config: onnx_config,
            label_perm,
        })
    }
}

impl SequenceClassifier for OnnxSequenceClassifier {
    /// Chunked batched inference.
    ///
    /// Phase 1: Tokenize ALL inputs outside any lock.
    /// Phase 2: Split into chunks of `config.batch_size`. For each chunk, pad
    ///          to that chunk's max sequence length and run one ORT forward
    ///          pass. Bounds peak activation allocation regardless of the
    ///          caller's input size.
    /// Phase 3: Concatenate logits in input order and post-process (softmax,
    ///          argmax) outside the lock.
    ///
    /// The whole-batch-in-one-run path was OOM-prone above ~150 pairs at
    /// `max_seq_len` near 1024 (see `RESEARCH_WRITEUP` §4.1). Chunking restores
    /// the documented invariant.
    #[allow(clippy::too_many_lines)]
    fn classify(
        &self,
        inputs: &[SequenceClassifierInput],
    ) -> Result<Vec<SequenceClassifierOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 1: Tokenize ALL inputs outside the lock
        let tokenized: Vec<_> = inputs
            .iter()
            .map(|input| {
                let start = Instant::now();
                let encoding = self
                    .tokenizer
                    .encode((input.text_a.as_str(), input.text_b.as_str()), true)
                    .map_err(|e| {
                        ModelError::inference_failed(format!("Tokenization failed: {e}"))
                    })?;

                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let type_ids = encoding.get_type_ids();
                let seq_len = ids.len().min(self.max_length);

                let input_ids: Vec<i64> = ids[..seq_len].iter().map(|&x| i64::from(x)).collect();
                let attention_mask: Vec<i64> =
                    mask[..seq_len].iter().map(|&x| i64::from(x)).collect();
                let token_type_ids: Vec<i64> =
                    type_ids[..seq_len].iter().map(|&x| i64::from(x)).collect();

                Ok((input_ids, attention_mask, token_type_ids, seq_len, start))
            })
            .collect::<Result<Vec<_>>>()?;

        let chunk_capacity = self.config.batch_size.max(1);
        let total = tokenized.len();
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(total);

        // Phase 2: Iterate chunks, each one a self-contained ORT forward pass.
        for chunk in tokenized.chunks(chunk_capacity) {
            let mut chunk_max_seq_len = 0;
            for (input_ids, _, _, _, _) in chunk {
                chunk_max_seq_len = chunk_max_seq_len.max(input_ids.len());
            }
            let chunk_size = chunk.len();
            let min_tokens = chunk.iter().map(|(ids, ..)| ids.len()).min().unwrap_or(0);
            debug!(
                chunk_size,
                max_tokens = chunk_max_seq_len,
                min_tokens,
                sessions = self.sessions.len(),
                "ONNX batching"
            );

            let mut batch_input_ids = Vec::with_capacity(chunk_size * chunk_max_seq_len);
            let mut batch_attention_mask = Vec::with_capacity(chunk_size * chunk_max_seq_len);
            let mut batch_token_type_ids = Vec::with_capacity(chunk_size * chunk_max_seq_len);

            for (input_ids, attention_mask, token_type_ids, _, _) in chunk {
                batch_input_ids.extend_from_slice(input_ids);
                batch_attention_mask.extend_from_slice(attention_mask);
                batch_token_type_ids.extend_from_slice(token_type_ids);

                let pad_len = chunk_max_seq_len - input_ids.len();
                batch_input_ids.extend(std::iter::repeat_n(0i64, pad_len));
                batch_attention_mask.extend(std::iter::repeat_n(0i64, pad_len));
                batch_token_type_ids.extend(std::iter::repeat_n(0i64, pad_len));
            }

            let batch_start = Instant::now();
            let chunk_logits = run_ort_batch_slice(
                &self.sessions[0],
                &batch_input_ids,
                &batch_attention_mask,
                &batch_token_type_ids,
                chunk_size,
                chunk_max_seq_len,
                self.has_token_type_ids,
            )?;
            let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
            debug!(
                chunk_size,
                max_tokens = chunk_max_seq_len,
                min_tokens,
                latency_ms = batch_ms,
                "NLI chunk inference"
            );

            all_logits.extend(chunk_logits);
        }

        // Phase 3: Post-process outside the lock
        let outputs = all_logits
            .into_iter()
            .zip(tokenized.iter())
            .map(|(logits, (_, _, _, _, start))| {
                let raw_probs = softmax(&logits);

                debug_assert!(
                    raw_probs.iter().all(|p| p.is_finite()),
                    "NaN or Inf detected in softmax output"
                );

                // Reorder model's raw probs into canonical
                // [entailment, neutral, contradiction] order.
                let probs: Vec<f64> = self
                    .label_perm
                    .iter()
                    .map(|&src| raw_probs.get(src).copied().unwrap_or(0.0))
                    .collect();

                let predicted_class = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx);

                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                SequenceClassifierOutput {
                    predicted_class,
                    predicted_label: self
                        .label_names
                        .get(predicted_class)
                        .cloned()
                        .unwrap_or_else(|| format!("class_{predicted_class}")),
                    probabilities: probs,
                    latency_ms,
                }
            })
            .collect();

        Ok(outputs)
    }

    fn device_info(&self) -> String {
        format!("ONNX-CPU (sessions={})", self.sessions.len())
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn label_map(&self) -> &[String] {
        &self.label_names
    }
}

// =============================================================================
// ONNX Embedding Backend
// =============================================================================

/// ONNX Runtime backend for sentence embedding models (e.g. all-MiniLM-L6-v2).
///
/// Mean-pools the last hidden state over non-padding tokens and L2-normalizes
/// the result to produce a fixed-size embedding vector.
pub struct OnnxEmbeddingBackend {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    max_length: usize,
    embedding_dim: usize,
    has_token_type_ids: bool,
}

impl OnnxEmbeddingBackend {
    /// Loads an ONNX embedding model from a directory (`model.onnx`).
    pub fn from_pretrained(model_path: &Path) -> Result<Self> {
        Self::load_session(model_path, "model.onnx")
    }

    /// Loads a quantized ONNX embedding model (`model_quantized.onnx`).
    pub fn from_pretrained_quantized(model_path: &Path) -> Result<Self> {
        Self::load_session(model_path, "model_quantized.onnx")
    }

    fn load_session(model_path: &Path, model_filename: &str) -> Result<Self> {
        let intra_threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);

        info!(
            "Loading ONNX embedding backend from: {} ({})",
            model_path.display(),
            model_filename
        );

        let model_file = model_path.join(model_filename);
        let session = Session::builder()
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .with_intra_threads(intra_threads)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?
            .commit_from_file(&model_file)
            .map_err(|e| ModelError::load_failed(&model_file, format!("{e}")))?;

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| ModelError::load_failed(&tokenizer_path, e.to_string()))?;

        let config_path = model_path.join("config.json");
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?,
        )?;

        #[allow(clippy::cast_possible_truncation)]
        let max_length = config["max_position_embeddings"].as_u64().unwrap_or(512) as usize;

        #[allow(clippy::cast_possible_truncation)]
        let embedding_dim = config["hidden_size"].as_u64().unwrap_or(384) as usize;

        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        info!(
            "ONNX embedding backend loaded: dim={}, max_length={}, token_type_ids={}",
            embedding_dim, max_length, has_token_type_ids
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            max_length,
            embedding_dim,
            has_token_type_ids,
        })
    }
}

/// Mean-pools hidden states over non-padding tokens and L2-normalizes the result.
#[allow(clippy::cast_precision_loss)]
fn mean_pool_and_normalize(
    hidden_data: &[f32],
    attn_mask_flat: &[i64],
    b: usize,
    seq_len: usize,
    max_seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut embedding = vec![0.0f32; hidden_dim];
    let mut count = 0usize;
    for s in 0..seq_len {
        if attn_mask_flat[b * max_seq_len + s] == 1 {
            let offset = (b * seq_len + s) * hidden_dim;
            for d in 0..hidden_dim {
                embedding[d] += hidden_data[offset + d];
            }
            count += 1;
        }
    }
    if count > 0 {
        let count_f = count as f32;
        for x in &mut embedding {
            *x /= count_f;
        }
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

impl EmbeddingSimilarity for OnnxEmbeddingBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        const BATCH_SIZE: usize = 32;

        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let batch = chunk.len();
            let mut max_seq_len = 0usize;
            let mut tokenized: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::with_capacity(batch);

            for text in chunk {
                let encoding = self.tokenizer.encode(*text, true).map_err(|e| {
                    ModelError::inference_failed(format!("Tokenization failed: {e}"))
                })?;
                let ids = encoding.get_ids();
                let seq_len = ids.len().min(self.max_length);
                max_seq_len = max_seq_len.max(seq_len);

                let input_ids: Vec<i64> = ids[..seq_len].iter().map(|&x| i64::from(x)).collect();
                let attn_mask: Vec<i64> = encoding.get_attention_mask()[..seq_len]
                    .iter()
                    .map(|&x| i64::from(x))
                    .collect();
                let type_ids: Vec<i64> = encoding.get_type_ids()[..seq_len]
                    .iter()
                    .map(|&x| i64::from(x))
                    .collect();
                tokenized.push((input_ids, attn_mask, type_ids));
            }

            let mut input_ids_flat: Vec<i64> = Vec::with_capacity(batch * max_seq_len);
            let mut attn_mask_flat: Vec<i64> = Vec::with_capacity(batch * max_seq_len);
            let mut type_ids_flat: Vec<i64> = Vec::with_capacity(batch * max_seq_len);

            for (mut ids, mut mask, mut types) in tokenized {
                ids.resize(max_seq_len, 0);
                mask.resize(max_seq_len, 0);
                types.resize(max_seq_len, 0);
                input_ids_flat.extend_from_slice(&ids);
                attn_mask_flat.extend_from_slice(&mask);
                type_ids_flat.extend_from_slice(&types);
            }

            let (hidden_data, seq_len, hidden_dim) = {
                let mut session_guard = self
                    .session
                    .lock()
                    .map_err(|e| ModelError::inference_failed(format!("Mutex lock failed: {e}")))?;

                let input_ids_tensor =
                    ort::value::Value::from_array(([batch, max_seq_len], input_ids_flat))
                        .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

                let attn_mask_tensor =
                    ort::value::Value::from_array(([batch, max_seq_len], attn_mask_flat.clone()))
                        .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

                let ort_inputs = if self.has_token_type_ids {
                    let type_ids_tensor =
                        ort::value::Value::from_array(([batch, max_seq_len], type_ids_flat))
                            .map_err(|e| ModelError::inference_failed(format!("{e}")))?;
                    ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attn_mask_tensor,
                        "token_type_ids" => type_ids_tensor,
                    ]
                } else {
                    ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attn_mask_tensor,
                    ]
                };

                let ort_outputs = session_guard
                    .run(ort_inputs)
                    .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

                // last_hidden_state shape: [batch, seq_len, hidden_dim]
                let tensor = ort_outputs["last_hidden_state"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ModelError::inference_failed(format!("{e}")))?;

                let shape = &tensor.0;
                let seq_len = usize::try_from(shape[1]).unwrap_or(max_seq_len);
                let hidden_dim = usize::try_from(shape[2]).unwrap_or(self.embedding_dim);
                let data: Vec<f32> = tensor.1.to_vec();
                (data, seq_len, hidden_dim)
            };

            for b in 0..batch {
                let embedding = mean_pool_and_normalize(
                    &hidden_data,
                    &attn_mask_flat,
                    b,
                    seq_len,
                    max_seq_len,
                    hidden_dim,
                );
                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn device_info(&self) -> String {
        "ONNX-CPU-Embedding".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for f32→f64 softmax (f32 arithmetic precision limit)
    const SOFTMAX_TOL: f64 = 1e-7;

    #[test]
    fn test_softmax_basic() {
        let probs = softmax(&[2.0, 1.0, 0.0]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < SOFTMAX_TOL);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_large_logits_no_overflow() {
        let probs = softmax(&[1000.0, 999.0, 998.0]);
        assert!(probs.iter().all(|p| p.is_finite()));
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < SOFTMAX_TOL);
    }

    #[test]
    fn test_softmax_negative_logits() {
        let probs = softmax(&[-1.0, -2.0, -3.0]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < SOFTMAX_TOL);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_equal_logits() {
        let probs = softmax(&[1.0, 1.0, 1.0]);
        // All probabilities should be ~0.333
        assert!(probs.iter().all(|p| (p - 1.0 / 3.0).abs() < SOFTMAX_TOL));
    }

    #[test]
    fn test_softmax_zero_logits() {
        let probs = softmax(&[0.0, 0.0, 0.0]);
        // All probabilities should be ~0.333
        assert!(probs.iter().all(|p| (p - 1.0 / 3.0).abs() < SOFTMAX_TOL));
    }

    #[test]
    fn test_softmax_single_element() {
        let probs = softmax(&[5.0]);
        assert!((probs[0] - 1.0).abs() < SOFTMAX_TOL);
    }

    #[test]
    fn test_softmax_two_elements() {
        let probs = softmax(&[0.0, 0.0]);
        assert!((probs[0] - 0.5).abs() < SOFTMAX_TOL);
        assert!((probs[1] - 0.5).abs() < SOFTMAX_TOL);
    }
}
