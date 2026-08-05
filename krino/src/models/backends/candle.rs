//! Candle inference backend.
//!
//! This module implements the `TokenClassifier` and `EmbeddingSimilarity` traits
//! using the Candle ML framework.
//! Uses `ModernBERT` architecture for `LettuceDetect` token classification
//! and BERT-based models for sentence embeddings.

use crate::error::{ModelError, Result};
use crate::models::inference::{
    EmbeddingSimilarity, TokenClassifier, TokenClassifierInput, TokenClassifierOutput,
    TokenPrediction,
};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Candle-based token classifier using `ModernBERT`.
///
/// This backend uses Candle's `ModernBERT` implementation for token classification.
/// It includes the proper classification head: dense -> GELU -> `LayerNorm` -> classifier.
pub struct CandleBackend {
    /// The underlying `ModernBERT` model
    model: ModernBert,

    /// Classification head dense layer
    head_dense: Linear,

    /// Classification head layer norm (no bias in `ModernBERT`)
    head_norm: LayerNorm,

    /// Final classifier layer
    classifier: Linear,

    /// Number of classification labels
    num_labels: usize,

    /// Device (CPU or CUDA)
    device: Device,

    /// Maximum sequence length
    max_length: usize,
}

impl CandleBackend {
    /// Creates a new Candle backend from a pre-trained `ModernBERT` model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory containing config, weights, and tokenizer
    /// * `num_labels` - Number of classification labels (2 for binary classification)
    ///
    /// # Errors
    ///
    /// Returns an error if the model files cannot be loaded or parsed.
    pub fn from_pretrained(model_path: &Path, num_labels: usize) -> Result<Self> {
        info!("Loading ModernBERT backend from: {}", model_path.display());

        // Determine device (prefer CUDA if available)
        let device = Self::get_device()?;
        debug!("Using device: {:?}", device);

        // Load config
        let config_path = model_path.join("config.json");
        let config: ModernBertConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?,
        )
        .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?;

        debug!(
            "Model config: hidden_size={}, num_hidden_layers={}",
            config.hidden_size, config.num_hidden_layers
        );

        let max_length = config.max_position_embeddings;
        let hidden_size = config.hidden_size;

        // Load weights using SafeTensors
        let weights_path = model_path.join("model.safetensors");
        let device_clone = device.clone();
        let vb = Self::load_safetensors(&weights_path, &device_clone)?;

        // Initialize ModernBERT model
        // The weights already have "model." prefix, so we pass vb directly
        let model = ModernBert::load(vb.clone(), &config)
            .map_err(|e| ModelError::load_failed(&weights_path, e.to_string()))?;

        // Load classification head: head.dense (no bias in `ModernBERT`)
        let head_dense =
            candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("head").pp("dense"))
                .map_err(|e| {
                    ModelError::load_failed(
                        &weights_path,
                        format!("Failed to load head.dense: {e}"),
                    )
                })?;

        // Load head layer norm (no bias in `ModernBERT`, so we need to create zero bias)
        let norm_weight = vb
            .pp("head")
            .pp("norm")
            .get(hidden_size, "weight")
            .map_err(|e| {
                ModelError::load_failed(
                    &weights_path,
                    format!("Failed to load head.norm.weight: {e}"),
                )
            })?;
        let norm_bias = Tensor::zeros(hidden_size, DType::F32, &device).map_err(|e| {
            ModelError::load_failed(
                &weights_path,
                format!("Failed to create zero bias for norm: {e}"),
            )
        })?;
        let head_norm = LayerNorm::new(norm_weight, norm_bias, config.layer_norm_eps);

        // Load final classifier
        let classifier =
            candle_nn::linear(hidden_size, num_labels, vb.pp("classifier")).map_err(|e| {
                ModelError::load_failed(&weights_path, format!("Failed to load classifier: {e}"))
            })?;

        info!("Successfully loaded ModernBERT backend with classification head");

        Ok(Self {
            model,
            head_dense,
            head_norm,
            classifier,
            num_labels,
            device,
            max_length,
        })
    }

    /// Determines the best available device (CUDA > Metal > CPU).
    #[allow(clippy::unnecessary_wraps)]
    fn get_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        if candle_core::utils::cuda_is_available() {
            return Ok(Device::new_cuda(0)
                .map_err(|e| ModelError::InitializationFailed(format!("CUDA error: {e}")))?);
        }

        #[cfg(feature = "metal")]
        if candle_core::utils::metal_is_available() {
            return Ok(Device::new_metal(0)
                .map_err(|e| ModelError::InitializationFailed(format!("Metal error: {e}")))?);
        }

        Ok(Device::Cpu)
    }

    /// Loads weights from a `SafeTensors` file.
    fn load_safetensors<'a>(path: &'a Path, device: &'a Device) -> Result<VarBuilder<'a>> {
        debug!("Loading weights from: {}", path.display());

        let data = std::fs::read(path).map_err(|e| ModelError::load_failed(path, e.to_string()))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| ModelError::load_failed(path, e.to_string()))?;

        // Convert to HashMap for from_tensors
        let mut tensor_map = std::collections::HashMap::new();
        for name in tensors.names() {
            let tensor_view = tensors.tensor(name).map_err(|e| {
                ModelError::load_failed(path, format!("Failed to get tensor {name}: {e}"))
            })?;

            let tensor = Tensor::from_raw_buffer(
                tensor_view.data(),
                tensor_view.dtype().try_into().map_err(|e| {
                    ModelError::load_failed(path, format!("Unsupported dtype for {name}: {e:?}"))
                })?,
                tensor_view.shape(),
                device,
            )
            .map_err(|e| {
                ModelError::load_failed(path, format!("Failed to create tensor for {name}: {e}"))
            })?;

            tensor_map.insert(name.clone(), tensor);
        }

        Ok(VarBuilder::from_tensors(tensor_map, DType::F32, device))
    }

    /// Runs forward pass through the model with proper classification head.
    ///
    /// Architecture: `ModernBERT` -> dense -> GELU -> `LayerNorm` -> classifier -> softmax
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Get encoder output from `ModernBERT`
        let encoder_output = self
            .model
            .forward(input_ids, attention_mask)
            .map_err(|e| ModelError::inference_failed(format!("ModernBERT forward failed: {e}")))?;

        // Apply classification head: dense projection
        let hidden = self
            .head_dense
            .forward(&encoder_output)
            .map_err(|e| ModelError::inference_failed(format!("Head dense forward failed: {e}")))?;

        // GELU activation
        let hidden = hidden
            .gelu()
            .map_err(|e| ModelError::inference_failed(format!("GELU activation failed: {e}")))?;

        // Layer normalization
        let hidden = self
            .head_norm
            .forward(&hidden)
            .map_err(|e| ModelError::inference_failed(format!("Head norm forward failed: {e}")))?;

        // Final classifier
        let logits = self
            .classifier
            .forward(&hidden)
            .map_err(|e| ModelError::inference_failed(format!("Classifier forward failed: {e}")))?;

        // Apply softmax to get probabilities
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| ModelError::inference_failed(format!("Softmax failed: {e}")))?;

        Ok(probs)
    }
}

impl TokenClassifier for CandleBackend {
    fn predict(&self, inputs: &[TokenClassifierInput]) -> Result<Vec<TokenClassifierOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();

        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            // Convert input_ids to tensor
            let input_ids_vec: Vec<u32> = input.input_ids.clone();
            let input_ids_tensor = Tensor::new(input_ids_vec.as_slice(), &self.device)
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to create input_ids tensor: {e}"))
                })?
                .unsqueeze(0)
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to unsqueeze input_ids: {e}"))
                })?;

            // Convert attention_mask to tensor
            let attention_mask_vec: Vec<u32> = input.attention_mask.clone();
            let attention_mask_tensor = Tensor::new(attention_mask_vec.as_slice(), &self.device)
                .map_err(|e| {
                    ModelError::inference_failed(format!(
                        "Failed to create attention_mask tensor: {e}"
                    ))
                })?
                .unsqueeze(0)
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to unsqueeze attention_mask: {e}"))
                })?;

            // Run forward pass
            let probs = self.forward(&input_ids_tensor, &attention_mask_tensor)?;

            // Remove batch dimension [1, seq_len, num_labels] -> [seq_len, num_labels]
            let probs = probs.squeeze(0).map_err(|e| {
                ModelError::inference_failed(format!("Failed to squeeze batch dimension: {e}"))
            })?;

            // Extract probabilities for each token
            let probs_vec = probs.to_vec2::<f32>().map_err(|e| {
                ModelError::inference_failed(format!("Failed to extract probabilities: {e}"))
            })?;

            let seq_len = input.input_ids.len();
            let mut tokens = Vec::with_capacity(seq_len);

            for (idx, &token_id) in input.input_ids.iter().enumerate() {
                if idx < probs_vec.len() {
                    let token_probs: Vec<f64> =
                        probs_vec[idx].iter().map(|&p| f64::from(p)).collect();

                    // Find predicted class (argmax)
                    let predicted_class = token_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map_or(0, |(idx, _)| idx);

                    tokens.push(TokenPrediction {
                        token_id,
                        token: format!("token_{token_id}"),
                        predicted_class,
                        probabilities: token_probs,
                        offset: input.offsets[idx],
                    });
                }
            }

            outputs.push(TokenClassifierOutput {
                tokens,
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }

        debug!(
            "Candle inference complete: {} inputs processed in {:.2}ms",
            inputs.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(outputs)
    }

    fn device_info(&self) -> String {
        format!("Candle ({:?})", self.device)
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Candle-based sentence embedding model.
///
/// Uses sentence-transformers (e.g., all-MiniLM-L6-v2) for computing
/// dense embeddings used in the groundedness pre-filtering step.
///
/// Architecture: BERT encoder -> mean pooling -> L2 normalization
pub struct CandleEmbeddingBackend {
    /// BERT model for encoding
    model: BertModel,

    /// Tokenizer for the embedding model
    tokenizer: Tokenizer,

    /// Device (CPU or CUDA)
    device: Device,

    /// Embedding dimensionality
    embedding_dim: usize,

    /// Maximum sequence length
    max_length: usize,
}

impl CandleEmbeddingBackend {
    /// Creates a new Candle embedding backend from a pre-trained sentence-transformers model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory containing config, weights, and tokenizer
    ///
    /// # Errors
    ///
    /// Returns an error if the model files cannot be loaded or parsed.
    pub fn from_pretrained(model_path: &Path) -> Result<Self> {
        info!(
            "Loading sentence-transformers embedding backend from: {}",
            model_path.display()
        );

        // Determine device
        let device = CandleBackend::get_device()?;
        debug!("Using device: {:?}", device);

        // Load config
        let config_path = model_path.join("config.json");
        let config: BertConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?,
        )
        .map_err(|e| ModelError::load_failed(&config_path, e.to_string()))?;

        debug!(
            "Embedding model config: hidden_size={}, num_hidden_layers={}",
            config.hidden_size, config.num_hidden_layers
        );

        let embedding_dim = config.hidden_size;
        let max_length = config.max_position_embeddings;

        // Load weights
        let weights_path = model_path.join("model.safetensors");
        let vb = CandleBackend::load_safetensors(&weights_path, &device)?;

        // Initialize BERT model
        let model = BertModel::load(vb, &config)
            .map_err(|e| ModelError::load_failed(&weights_path, e.to_string()))?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| ModelError::load_failed(&tokenizer_path, e.to_string()))?;

        info!(
            "Successfully loaded sentence-transformers embedding backend (dim={})",
            embedding_dim
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_dim,
            max_length,
        })
    }

    /// Mean pooling over token embeddings, masked by attention mask.
    ///
    /// This produces a single fixed-size embedding per sequence by averaging
    /// all token embeddings, weighted by the attention mask (to ignore padding).
    fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match embedding dimensions
        // attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to unsqueeze attention mask: {e}"))
            })?
            .expand(embeddings.dims())
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to expand attention mask: {e}"))
            })?
            .to_dtype(embeddings.dtype())
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to convert mask dtype: {e}"))
            })?;

        // Multiply embeddings by mask and sum over sequence dimension
        let sum_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| ModelError::inference_failed(format!("Failed to mask embeddings: {e}")))?
            .sum(1)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to sum masked embeddings: {e}"))
            })?;

        // Sum the mask to get the count of non-padding tokens
        let sum_mask = mask_expanded
            .sum(1)
            .map_err(|e| ModelError::inference_failed(format!("Failed to sum mask: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| ModelError::inference_failed(format!("Failed to clamp mask sum: {e}")))?;

        // Divide to get the mean
        Ok(sum_embeddings.div(&sum_mask).map_err(|e| {
            ModelError::inference_failed(format!("Failed to compute mean pooling: {e}"))
        })?)
    }
}

impl EmbeddingSimilarity for CandleEmbeddingBackend {
    #[allow(clippy::too_many_lines)]
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Process in batches for better performance (batch_size=32)
        let batch_size = 32;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(batch_size) {
            // Phase 1: Tokenize all texts in the batch
            let mut tokenized = Vec::with_capacity(chunk.len());
            let mut max_seq_len = 0;

            for text in chunk {
                let encoding = self.tokenizer.encode(*text, true).map_err(|e| {
                    ModelError::inference_failed(format!("Tokenization failed: {e}"))
                })?;

                let ids = encoding.get_ids();
                let seq_len = ids.len().min(self.max_length);
                max_seq_len = max_seq_len.max(seq_len);

                tokenized.push((
                    ids[..seq_len].to_vec(),
                    encoding.get_attention_mask()[..seq_len].to_vec(),
                    encoding.get_type_ids()[..seq_len].to_vec(),
                ));
            }

            // Phase 2: Pad all sequences to max_seq_len in batch
            for (input_ids, attention_mask, token_type_ids) in &mut tokenized {
                while input_ids.len() < max_seq_len {
                    input_ids.push(0); // Pad with 0
                    attention_mask.push(0);
                    token_type_ids.push(0);
                }
            }

            // Phase 3: Convert to batch tensors [batch_size, max_seq_len]
            let batch_size_actual = tokenized.len();
            let mut input_ids_data: Vec<i64> = Vec::with_capacity(batch_size_actual * max_seq_len);
            let mut attention_mask_data: Vec<i64> =
                Vec::with_capacity(batch_size_actual * max_seq_len);
            let mut token_type_ids_data: Vec<i64> =
                Vec::with_capacity(batch_size_actual * max_seq_len);

            for (input_ids, attention_mask, token_type_ids) in &tokenized {
                for id in input_ids {
                    input_ids_data.push(i64::from(*id));
                }
                for mask in attention_mask {
                    attention_mask_data.push(i64::from(*mask));
                }
                for type_id in token_type_ids {
                    token_type_ids_data.push(i64::from(*type_id));
                }
            }

            // Convert to tensors
            let input_ids_tensor = Tensor::new(input_ids_data.as_slice(), &self.device)
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to create input_ids tensor: {e}"))
                })?
                .reshape((batch_size_actual, max_seq_len))
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to reshape input_ids: {e}"))
                })?;

            let attention_mask_tensor = Tensor::new(attention_mask_data.as_slice(), &self.device)
                .map_err(|e| {
                    ModelError::inference_failed(format!(
                        "Failed to create attention_mask tensor: {e}"
                    ))
                })?
                .reshape((batch_size_actual, max_seq_len))
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to reshape attention_mask: {e}"))
                })?;

            let token_type_ids_tensor = Tensor::new(token_type_ids_data.as_slice(), &self.device)
                .map_err(|e| {
                    ModelError::inference_failed(format!(
                        "Failed to create token_type_ids tensor: {e}"
                    ))
                })?
                .reshape((batch_size_actual, max_seq_len))
                .map_err(|e| {
                    ModelError::inference_failed(format!("Failed to reshape token_type_ids: {e}"))
                })?;

            // Phase 4: Forward pass through BERT (batched)
            let embeddings = self
                .model
                .forward(
                    &input_ids_tensor,
                    &token_type_ids_tensor,
                    Some(&attention_mask_tensor),
                )
                .map_err(|e| ModelError::inference_failed(format!("BERT forward failed: {e}")))?;

            // Phase 5: Mean pooling (batched)
            let pooled = Self::mean_pool(&embeddings, &attention_mask_tensor)?;

            // Phase 6: L2 normalization (batched)
            let norm = pooled
                .sqr()
                .map_err(|e| ModelError::inference_failed(format!("Failed to square: {e}")))?
                .sum(1)
                .map_err(|e| ModelError::inference_failed(format!("Failed to sum squares: {e}")))?
                .sqrt()
                .map_err(|e| ModelError::inference_failed(format!("Failed to sqrt: {e}")))?;

            let normalized = pooled
                .broadcast_div(&norm.unsqueeze(1).map_err(|e| {
                    ModelError::inference_failed(format!("Failed to unsqueeze norm: {e}"))
                })?)
                .map_err(|e| ModelError::inference_failed(format!("Failed to normalize: {e}")))?;

            // Phase 7: Convert to individual embeddings
            for i in 0..batch_size_actual {
                let vec: Vec<f32> = normalized
                    .get(i)
                    .map_err(|e| {
                        ModelError::inference_failed(format!("Failed to index normalized: {e}"))
                    })?
                    .to_vec1()
                    .map_err(|e| {
                        ModelError::inference_failed(format!("Failed to convert to vec: {e}"))
                    })?;
                all_embeddings.push(vec);
            }
        }

        debug!(
            "Computed {} embeddings (dim={}) in batches of {}",
            all_embeddings.len(),
            self.embedding_dim,
            batch_size
        );

        Ok(all_embeddings)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn device_info(&self) -> String {
        format!("Candle-Embedding ({:?})", self.device)
    }
}
