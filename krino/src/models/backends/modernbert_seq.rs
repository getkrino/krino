//! `ModernBERT` sequence classification backend for NLI.
//!
//! This module implements sequence classification using `ModernBERT` for Natural Language Inference (NLI).
//! It classifies premise-hypothesis pairs into three categories: entailment, neutral, or contradiction.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use std::path::Path;
use tracing::{debug, info};

/// Sequence classification output with 3-class probabilities.
#[derive(Debug, Clone)]
pub struct NliOutput {
    /// Probability of entailment (premise implies hypothesis)
    pub entailment: f64,

    /// Probability of neutral (insufficient information)
    pub neutral: f64,

    /// Probability of contradiction (premise contradicts hypothesis)
    pub contradiction: f64,

    /// Predicted class (0=entailment, 1=neutral, 2=contradiction)
    pub predicted_class: usize,
}

/// `ModernBERT`-based NLI classifier.
///
/// This backend uses `ModernBERT` with a sequence classification head for Natural Language Inference.
/// It processes premise-hypothesis pairs and returns entailment/neutral/contradiction predictions.
pub struct ModernBertSeqBackend {
    /// The underlying `ModernBERT` model
    model: ModernBert,

    /// Classification head
    classifier: Linear,

    /// Number of classes (3 for NLI: entailment, neutral, contradiction)
    num_labels: usize,

    /// Device (CPU or CUDA)
    device: Device,

    /// Maximum sequence length
    max_length: usize,
}

impl ModernBertSeqBackend {
    /// Creates a new `ModernBERT` sequence classification backend from a pre-trained model.
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
            "Loading ModernBERT sequence classification backend from: {}",
            model_path.display()
        );

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
        let num_labels = 3; // NLI: entailment, neutral, contradiction

        // Load weights using SafeTensors
        let weights_path = model_path.join("model.safetensors");
        let device_clone = device.clone();
        let vb = Self::load_safetensors(&weights_path, &device_clone)?;

        // Initialize ModernBERT model
        let model = ModernBert::load(vb.clone(), &config)
            .map_err(|e| ModelError::load_failed(&weights_path, e.to_string()))?;

        // Load classification head
        let classifier =
            candle_nn::linear(hidden_size, num_labels, vb.pp("classifier")).map_err(|e| {
                ModelError::load_failed(&weights_path, format!("Failed to load classifier: {e}"))
            })?;

        info!("Successfully loaded ModernBERT sequence classification backend");

        Ok(Self {
            model,
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

    /// Runs forward pass for sequence classification with mean pooling.
    ///
    /// Architecture: `ModernBERT` -> mean pool -> classifier -> softmax
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Get encoder output from ModernBERT [batch, seq_len, hidden_size]
        let encoder_output = self
            .model
            .forward(input_ids, attention_mask)
            .map_err(|e| ModelError::inference_failed(format!("ModernBERT forward failed: {e}")))?;

        // Apply mean pooling over sequence length
        // We need to account for the attention mask to only average over real tokens
        let pooled = Self::mean_pool(&encoder_output, attention_mask)?;

        // Apply classifier head
        let logits = self
            .classifier
            .forward(&pooled)
            .map_err(|e| ModelError::inference_failed(format!("Classifier forward failed: {e}")))?;

        // Apply softmax to get probabilities
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| ModelError::inference_failed(format!("Softmax failed: {e}")))?;

        Ok(probs)
    }

    /// Mean pooling over sequence length, accounting for attention mask.
    fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch, seq_len, hidden_size]
        // attention_mask: [batch, seq_len]

        // Expand attention mask to [batch, seq_len, hidden_size]
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| ModelError::inference_failed(format!("Failed to unsqueeze mask: {e}")))?
            .broadcast_as(hidden_states.shape())
            .map_err(|e| ModelError::inference_failed(format!("Failed to broadcast mask: {e}")))?
            .to_dtype(hidden_states.dtype())
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to convert mask dtype: {e}"))
            })?;

        // Multiply hidden states by mask
        let masked = hidden_states.mul(&mask_expanded).map_err(|e| {
            ModelError::inference_failed(format!("Failed to mask hidden states: {e}"))
        })?;

        // Sum over sequence length
        let sum = masked.sum(1).map_err(|e| {
            ModelError::inference_failed(format!("Failed to sum masked states: {e}"))
        })?;

        // Count number of tokens (sum of attention mask)
        let count = attention_mask
            .sum(1)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to sum attention mask: {e}"))
            })?
            .unsqueeze(1)
            .map_err(|e| ModelError::inference_failed(format!("Failed to unsqueeze count: {e}")))?;

        // Divide by count to get mean
        let mean = sum
            .div(&count)
            .map_err(|e| ModelError::inference_failed(format!("Failed to compute mean: {e}")))?;

        Ok(mean)
    }

    /// Classifies a premise-hypothesis pair.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs for premise + hypothesis
    /// * `attention_mask` - Attention mask
    ///
    /// # Returns
    ///
    /// `NliOutput` with probabilities and predicted class.
    ///
    /// # Panics
    ///
    /// Panics if probability comparison fails (should never happen with valid f64 values).
    pub fn classify(&self, input_ids: &[u32], attention_mask: &[u32]) -> Result<NliOutput> {
        // Convert to tensors
        let input_ids_tensor = Tensor::new(input_ids, &self.device)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to create input_ids tensor: {e}"))
            })?
            .unsqueeze(0)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to unsqueeze input_ids: {e}"))
            })?;

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to create attention_mask tensor: {e}"))
            })?
            .unsqueeze(0)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to unsqueeze attention_mask: {e}"))
            })?;

        // Run forward pass
        let probs = self.forward(&input_ids_tensor, &attention_mask_tensor)?;

        // Extract probabilities [batch=1, num_labels=3]
        let probs_vec = probs
            .squeeze(0)
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to squeeze batch dimension: {e}"))
            })?
            .to_vec1::<f32>()
            .map_err(|e| {
                ModelError::inference_failed(format!("Failed to extract probabilities: {e}"))
            })?;

        // Find predicted class (argmax)
        let predicted_class = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx);

        Ok(NliOutput {
            entailment: f64::from(probs_vec[0]),
            neutral: f64::from(probs_vec[1]),
            contradiction: f64::from(probs_vec[2]),
            predicted_class,
        })
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Returns device information.
    #[must_use]
    pub fn device_info(&self) -> String {
        format!("ModernBERT Sequence Classification ({:?})", self.device)
    }
}
