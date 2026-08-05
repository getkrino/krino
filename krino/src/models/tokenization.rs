//! Tokenization utilities with offset mapping.
//!
//! This module provides helpers for tokenizing text while preserving character offsets,
//! which is essential for mapping token-level predictions back to spans in the original text.

use crate::error::{ModelError, Result};
use crate::models::inference::TokenClassifierInput;
use tokenizers::{Encoding, Tokenizer};
use tracing::debug;

/// Tokenizes text and creates a `TokenClassifierInput` with offset mapping.
///
/// # Arguments
///
/// * `tokenizer` - `HuggingFace` tokenizer instance
/// * `text` - Text to tokenize
/// * `add_special_tokens` - Whether to add special tokens (CLS, SEP)
/// * `max_length` - Maximum sequence length (will truncate if exceeded)
///
/// # Returns
///
/// A `TokenClassifierInput` with token IDs, attention mask, and character offsets.
///
/// # Errors
///
/// Returns an error if tokenization fails.
///
/// # Examples
///
/// ```rust,ignore
/// use krino::models::tokenization::tokenize_with_offsets;
/// use tokenizers::Tokenizer;
///
/// let tokenizer = Tokenizer::from_file("tokenizer.json")?;
/// let input = tokenize_with_offsets(&tokenizer, "Hello world", true, 512)?;
/// assert!(!input.input_ids.is_empty());
/// assert_eq!(input.input_ids.len(), input.offsets.len());
/// ```
pub fn tokenize_with_offsets(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
    max_length: usize,
) -> Result<TokenClassifierInput> {
    debug!("Tokenizing text: {} chars", text.len());

    // Encode with offset mapping enabled
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|e| ModelError::inference_failed(format!("Tokenization failed: {e}")))?;

    // Extract token IDs
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Extract attention mask (1 for real tokens, 0 for padding)
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    // Extract character offsets
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

    // Truncate if needed
    let (input_ids, attention_mask, offsets) = if input_ids.len() > max_length {
        debug!(
            "Truncating sequence from {} to {} tokens",
            input_ids.len(),
            max_length
        );
        (
            input_ids[..max_length].to_vec(),
            attention_mask[..max_length].to_vec(),
            offsets[..max_length].to_vec(),
        )
    } else {
        (input_ids, attention_mask, offsets)
    };

    debug!(
        "Tokenization complete: {} tokens, {} offsets",
        input_ids.len(),
        offsets.len()
    );

    Ok(TokenClassifierInput {
        input_ids,
        attention_mask,
        offsets,
        token_type_ids: None,
        text: text.to_string(),
    })
}

/// Tokenizes multiple texts in batch.
///
/// # Arguments
///
/// * `tokenizer` - `HuggingFace` tokenizer instance
/// * `texts` - Texts to tokenize
/// * `add_special_tokens` - Whether to add special tokens (CLS, SEP)
/// * `max_length` - Maximum sequence length per text
///
/// # Returns
///
/// A vector of `TokenClassifierInput` instances.
///
/// # Errors
///
/// Returns an error if any tokenization fails.
pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    texts: &[impl AsRef<str>],
    add_special_tokens: bool,
    max_length: usize,
) -> Result<Vec<TokenClassifierInput>> {
    texts
        .iter()
        .map(|text| tokenize_with_offsets(tokenizer, text.as_ref(), add_special_tokens, max_length))
        .collect()
}

/// Decodes token IDs back to text.
///
/// # Arguments
///
/// * `tokenizer` - `HuggingFace` tokenizer instance
/// * `token_ids` - Token IDs to decode
/// * `skip_special_tokens` - Whether to skip special tokens in output
///
/// # Returns
///
/// Decoded text string.
///
/// # Errors
///
/// Returns an error if decoding fails.
pub fn decode_tokens(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    tokenizer
        .decode(token_ids, skip_special_tokens)
        .map_err(|e| ModelError::inference_failed(format!("Decoding failed: {e}")).into())
}

/// Merges consecutive tokens with the same predicted class into spans.
///
/// This is used to convert token-level predictions into character spans for the final report.
///
/// # Arguments
///
/// * `tokens` - Token predictions with offsets
/// * `target_class` - Class to extract spans for (e.g., 1 for hallucinated)
///
/// # Returns
///
/// A vector of `(start, end, confidence)` tuples representing character spans.
#[must_use]
#[allow(clippy::cast_precision_loss)]
///
/// # Examples
///
/// ```rust
/// use krino::models::inference::TokenPrediction;
/// use krino::models::tokenization::merge_token_spans;
///
/// let tokens = vec![
///     TokenPrediction {
///         token_id: 101,
///         token: "[CLS]".to_string(),
///         predicted_class: 0,
///         probabilities: vec![0.9, 0.1],
///         offset: (0, 0),
///     },
///     TokenPrediction {
///         token_id: 1996,
///         token: "The".to_string(),
///         predicted_class: 1,
///         probabilities: vec![0.2, 0.8],
///         offset: (0, 3),
///     },
///     TokenPrediction {
///         token_id: 2194,
///         token: "company".to_string(),
///         predicted_class: 1,
///         probabilities: vec![0.3, 0.7],
///         offset: (4, 11),
///     },
/// ];
///
/// let spans = merge_token_spans(&tokens, 1);
/// assert_eq!(spans.len(), 1);
/// assert_eq!(spans[0], (0, 11, 0.75)); // Merged span with average confidence
/// ```
pub fn merge_token_spans(
    tokens: &[crate::models::inference::TokenPrediction],
    target_class: usize,
) -> Vec<(usize, usize, f64)> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_end: usize = 0;
    let mut current_confidences = Vec::new();

    for token in tokens {
        // Skip special tokens (zero-length offsets)
        if token.offset.0 == token.offset.1 {
            continue;
        }

        // Check if token was classified as the target class
        let is_target = token.predicted_class == target_class;

        // Get probability for confidence scoring
        let prob = token
            .probabilities
            .get(target_class)
            .copied()
            .unwrap_or(0.0);

        if is_target {
            // Token belongs to target class
            if let Some(start) = current_start {
                // Continue existing span
                current_end = token.offset.1;
                current_confidences.push(prob);
            } else {
                // Start new span
                current_start = Some(token.offset.0);
                current_end = token.offset.1;
                current_confidences.push(prob);
            }
        } else if let Some(start) = current_start {
            // End current span
            let avg_confidence =
                current_confidences.iter().sum::<f64>() / current_confidences.len() as f64;
            spans.push((start, current_end, avg_confidence));
            current_start = None;
            current_confidences.clear();
        }
    }

    // Handle final span if still open
    if let Some(start) = current_start {
        let avg_confidence =
            current_confidences.iter().sum::<f64>() / current_confidences.len() as f64;
        spans.push((start, current_end, avg_confidence));
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inference::TokenPrediction;

    #[test]
    fn test_merge_token_spans_single_token() {
        let tokens = vec![TokenPrediction {
            token_id: 1,
            token: "test".to_string(),
            predicted_class: 1,
            probabilities: vec![0.2, 0.8],
            offset: (0, 4),
        }];

        let spans = merge_token_spans(&tokens, 1);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], (0, 4, 0.8));
    }

    #[test]
    fn test_merge_token_spans_consecutive() {
        let tokens = vec![
            TokenPrediction {
                token_id: 1,
                token: "The".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (0, 3),
            },
            TokenPrediction {
                token_id: 2,
                token: "company".to_string(),
                predicted_class: 1,
                probabilities: vec![0.3, 0.7],
                offset: (4, 11),
            },
        ];

        let spans = merge_token_spans(&tokens, 1);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], (0, 11, 0.75)); // Average of 0.8 and 0.7
    }

    #[test]
    fn test_merge_token_spans_separated() {
        let tokens = vec![
            TokenPrediction {
                token_id: 1,
                token: "The".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (0, 3),
            },
            TokenPrediction {
                token_id: 2,
                token: "is".to_string(),
                predicted_class: 0,
                probabilities: vec![0.9, 0.1],
                offset: (4, 6),
            },
            TokenPrediction {
                token_id: 3,
                token: "wrong".to_string(),
                predicted_class: 1,
                probabilities: vec![0.1, 0.9],
                offset: (7, 12),
            },
        ];

        let spans = merge_token_spans(&tokens, 1);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], (0, 3, 0.8));
        assert_eq!(spans[1], (7, 12, 0.9));
    }

    #[test]
    fn test_merge_token_spans_skip_special_tokens() {
        let tokens = vec![
            TokenPrediction {
                token_id: 101,
                token: "[CLS]".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (0, 0), // Zero-length offset
            },
            TokenPrediction {
                token_id: 1,
                token: "test".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (0, 4),
            },
            TokenPrediction {
                token_id: 102,
                token: "[SEP]".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (4, 4), // Zero-length offset
            },
        ];

        let spans = merge_token_spans(&tokens, 1);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], (0, 4, 0.8)); // Special tokens skipped
    }

    #[test]
    fn test_merge_token_spans_no_matches() {
        let tokens = vec![TokenPrediction {
            token_id: 1,
            token: "test".to_string(),
            predicted_class: 0,
            probabilities: vec![0.8, 0.2],
            offset: (0, 4),
        }];

        let spans = merge_token_spans(&tokens, 1);
        assert!(spans.is_empty());
    }
}
