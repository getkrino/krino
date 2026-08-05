//! Generic inference abstractions for ML models.
//!
//! This module defines backend-agnostic traits and types for running inference
//! with ML models. Specific backends (Candle, ONNX, etc.) implement these traits.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Trait for token-level classification models.
///
/// Implementors provide token classification predictions (e.g., hallucination
/// detection, NER, POS tagging) for input sequences.
///
/// # Determinism
///
/// All implementations MUST be deterministic: same input always produces
/// the same output with identical probabilities.
pub trait TokenClassifier: Send + Sync {
    /// Runs inference on a batch of inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of tokenized inputs
    ///
    /// # Returns
    ///
    /// Predictions for each token in each sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    fn predict(&self, inputs: &[TokenClassifierInput]) -> Result<Vec<TokenClassifierOutput>>;

    /// Returns the model's device (CPU, CUDA, etc.) as a string.
    fn device_info(&self) -> String;

    /// Returns the maximum sequence length supported by the model.
    fn max_length(&self) -> usize;
}

/// Input for token classification.
///
/// Contains tokenized text with metadata needed for inference and
/// post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClassifierInput {
    /// Token IDs
    pub input_ids: Vec<u32>,

    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,

    /// Character offsets for each token (start, end)
    ///
    /// Used to map token predictions back to character positions in the
    /// original text.
    pub offsets: Vec<(usize, usize)>,

    /// Token type IDs (optional, for models like BERT)
    ///
    /// 0 for first segment, 1 for second segment
    pub token_type_ids: Option<Vec<u32>>,

    /// Original text (for reference/debugging)
    pub text: String,
}

/// Output from token classification.
///
/// Contains per-token predictions with probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClassifierOutput {
    /// Per-token predictions
    ///
    /// One prediction per token in the input sequence.
    pub tokens: Vec<TokenPrediction>,

    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Prediction for a single token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPrediction {
    /// Token ID
    pub token_id: u32,

    /// Token text (decoded)
    pub token: String,

    /// Predicted class (e.g., 0=supported, 1=hallucinated)
    pub predicted_class: usize,

    /// Probability distribution over all classes
    ///
    /// For binary classification: [`p(class_0)`, `p(class_1)`]
    pub probabilities: Vec<f64>,

    /// Character offset in original text (start, end)
    pub offset: (usize, usize),
}

impl TokenPrediction {
    /// Returns the confidence (probability) of the predicted class.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.probabilities
            .get(self.predicted_class)
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns true if this token is predicted as the given class.
    #[must_use]
    pub fn is_class(&self, class: usize) -> bool {
        self.predicted_class == class
    }
}

/// Configuration for tokenization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maximum sequence length
    pub max_length: usize,

    /// Truncation strategy
    pub truncation: TruncationStrategy,

    /// Padding strategy
    pub padding: PaddingStrategy,

    /// Whether to add special tokens ([CLS], [SEP])
    pub add_special_tokens: bool,
}

/// Truncation strategy for sequences exceeding `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    /// Don't truncate
    DoNotTruncate,

    /// Truncate the longest sequence
    LongestFirst,

    /// Truncate only the first sequence
    OnlyFirst,

    /// Truncate only the second sequence
    OnlySecond,
}

/// Padding strategy for sequences shorter than `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddingStrategy {
    /// Don't pad
    DoNotPad,

    /// Pad to the longest sequence in the batch
    Longest,

    /// Pad to `max_length`
    MaxLength,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: 4096,
            truncation: TruncationStrategy::OnlyFirst,
            padding: PaddingStrategy::Longest,
            add_special_tokens: true,
        }
    }
}

/// Trait for sequence-level classificaiton models (NLI, sentiment, etc.).
///
/// `SequenceClassifier` produces a single classification per input pair.
///
/// # Determinism
///
/// All implementations must be deterministic.
pub trait SequenceClassifier: Send + Sync {
    /// Classifies a batch of input pairs.
    ///
    /// For NLI: each input is a (premise, hypothesis) pair.
    /// Returns one classification per input
    fn classify(&self, inputs: &[SequenceClassifierInput])
    -> Result<Vec<SequenceClassifierOutput>>;

    /// Returns device info string.
    fn device_info(&self) -> String;

    /// Returns the maximim total sequence length (premise + hypothesis + special tokens).
    fn max_length(&self) -> usize;

    /// Returns the label mapping (index -> label name).
    fn label_map(&self) -> &[String];
}

/// Trait for computing text embeddings for similarity comparison.
///
/// Used by the groundedness module for pre-filtering context sentences
/// before running the more expensive NLI model.
///
/// # Determinism
///
/// All implementations MUST be deterministic: same input always produces
/// the same output embeddings.
pub trait EmbeddingSimilarity: Send + Sync {
    /// Computes embeddings for a batch of texts.
    ///
    /// Returns one embedding vector per input text.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to embed
    ///
    /// # Returns
    ///
    /// Vector of embedding vectors, one per input text.
    /// Each embedding is a dense vector of f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Returns the embedding dimensionality.
    fn embedding_dim(&self) -> usize;

    /// Returns device info string.
    fn device_info(&self) -> String;
}

/// Input for sequence classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceClassifierInput {
    /// First segment (premise / context for NLI)
    pub text_a: String,

    /// Second segment (hypothesis / claim for NLI)
    pub text_b: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceClassifierOutput {
    /// Predict class index
    pub predicted_class: usize,

    /// Predicted class label (e.g., "entailment")
    pub predicted_label: String,

    /// Probability distribution over all classes
    pub probabilities: Vec<f64>,

    /// Inference latency (ms)
    pub latency_ms: f64,
}

/// Computes cosine similarity between two embedding vectors.
///
/// Cosine similarity is defined as the dot product of the normalized vectors:
/// `sim(a, b) = (a · b) / (||a|| * ||b||)`
///
/// Returns a value in [-1, 1]:
/// - 1.0: vectors point in the same direction (identical)
/// - 0.0: vectors are orthogonal (unrelated)
/// - -1.0: vectors point in opposite directions (opposite)
///
/// # Determinism
///
/// This is a pure function. Same inputs always produce the same output.
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Panics
///
/// Panics in debug mode if the vectors have different lengths.
///
/// # Examples
///
/// ```
/// use krino::models::inference::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
///
/// let c = vec![1.0, 0.0];
/// let d = vec![0.0, 1.0];
/// assert!(cosine_similarity(&c, &d).abs() < 1e-6); // orthogonal
/// ```
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Computes a cosine similarity matrix between two sets of embeddings.
///
/// This is significantly faster than computing pairwise similarities one-by-one
/// because it uses matrix multiplication to compute all dot products at once.
///
/// # Arguments
///
/// * `a_embeddings` - First set of embeddings (N vectors)
/// * `b_embeddings` - Second set of embeddings (M vectors)
///
/// # Returns
///
/// An N×M matrix where `result[i][j]` is the cosine similarity between
/// `a_embeddings[i]` and `b_embeddings[j]`.
///
/// # Performance
///
/// For N=20, M=900, dim=384:
/// - Naive approach: ~18,000 dot products computed individually
/// - Matrix approach: Single matrix multiply (20×384) @ (384×900)
/// - Speedup: ~10-50× depending on BLAS implementation
///
/// # Examples
///
/// ```
/// use krino::models::inference::cosine_similarity_matrix;
///
/// let claims = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let contexts = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
/// let matrix = cosine_similarity_matrix(&claims, &contexts);
///
/// assert_eq!(matrix.len(), 2); // 2 claims
/// assert_eq!(matrix[0].len(), 2); // 2 contexts per claim
/// assert!((matrix[0][0] - 1.0).abs() < 1e-6); // claim[0] identical to context[0]
/// ```
///
/// # Panics
/// Panics if embeddings are empty or have mismatched dimensions.
#[must_use]
pub fn cosine_similarity_matrix(
    a_embeddings: &[Vec<f32>],
    b_embeddings: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    use ndarray::{Array2, Axis};

    if a_embeddings.is_empty() || b_embeddings.is_empty() {
        return vec![vec![]; a_embeddings.len()];
    }

    let n = a_embeddings.len();
    let m = b_embeddings.len();
    let dim = a_embeddings[0].len();

    // Build matrices from embeddings
    // A: [N, dim], B: [M, dim]
    let mut a_data = Vec::with_capacity(n * dim);
    let mut b_data = Vec::with_capacity(m * dim);

    for emb in a_embeddings {
        a_data.extend_from_slice(emb);
    }
    for emb in b_embeddings {
        b_data.extend_from_slice(emb);
    }

    let a_matrix = Array2::from_shape_vec((n, dim), a_data).expect("Invalid shape for A");
    let b_matrix = Array2::from_shape_vec((m, dim), b_data).expect("Invalid shape for B");

    // Compute L2 norms for each row
    let a_norms = a_matrix.map_axis(Axis(1), |row| row.mapv(|x| x * x).sum().sqrt());
    let b_norms = b_matrix.map_axis(Axis(1), |row| row.mapv(|x| x * x).sum().sqrt());

    // Normalize rows
    let a_normalized = &a_matrix / &a_norms.insert_axis(Axis(1));
    let b_normalized = &b_matrix / &b_norms.insert_axis(Axis(1));

    // Compute similarity matrix: A_normalized @ B_normalized^T
    // Result: [N, M]
    let similarity_matrix = a_normalized.dot(&b_normalized.t());

    // Convert back to Vec<Vec<f32>>
    similarity_matrix
        .outer_iter()
        .map(|row| row.to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::float_cmp)] // Allow exact float comparisons in tests
    #[test]
    fn test_token_prediction_confidence() {
        let pred = TokenPrediction {
            token_id: 101,
            token: "hello".to_string(),
            predicted_class: 1,
            probabilities: vec![0.3, 0.7],
            offset: (0, 5),
        };

        assert_eq!(pred.confidence(), 0.7);
        assert!(pred.is_class(1));
        assert!(!pred.is_class(0));
    }

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert_eq!(config.max_length, 4096);
        assert_eq!(config.truncation, TruncationStrategy::OnlyFirst);
        assert!(config.add_special_tokens);
    }

    #[test]
    fn test_truncation_strategy_serialization() {
        let strategy = TruncationStrategy::OnlyFirst;
        let json = serde_json::to_string(&strategy).unwrap();
        assert_eq!(json, "\"only_first\"");
    }

    // --- Cosine similarity tests ---

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 0.0];
        // Cosine of 45 degrees = 1/sqrt(2)
        let sim = cosine_similarity(&a, &b);
        assert!((sim - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_deterministic() {
        let a = vec![0.5, 0.3, 0.8, 0.2];
        let b = vec![0.7, 0.4, 0.6, 0.3];

        let sim1 = cosine_similarity(&a, &b);
        let sim2 = cosine_similarity(&a, &b);

        assert_eq!(sim1, sim2, "Cosine similarity must be deterministic");
    }
}
