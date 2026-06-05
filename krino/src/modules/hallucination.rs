//! Token-level hallucination detection module.
//!
//! This module implements hallucination detection using a fine-tuned BERT-style model
//! that classifies each token as either supported or hallucinated. It's based on the
//! `LettuceDetect` architecture.
//!
//! # Approach
//!
//! 1. **Tokenization**: Input text is tokenized with offset mapping preserved
//! 2. **Classification**: Each token is classified as 0 (supported) or 1 (hallucinated)
//! 3. **Span Extraction**: Consecutive hallucinated tokens are merged into spans
//! 4. **Chunking**: Long texts are split into overlapping chunks to handle sequences
//!    exceeding the model's maximum length
//! 5. **Aggregation**: Chunk-level predictions are aggregated using max pooling
//!
//! # Example
//!
//! ```rust,ignore
//! use krino::modules::hallucination::HallucinationDetector;
//! use krino::models::backends::CandleBackend;
//!
//! let backend = CandleBackend::from_pretrained(model_path, 2)?;
//! let detector = HallucinationDetector::new(backend);
//!
//! let result = detector.detect("The company was founded in 1987.")?;
//! for span in result.hallucinated_spans {
//!     println!("Hallucinated: '{}' at {:?}", span.text, (span.start, span.end));
//! }
//! ```

use crate::error::Result;
use crate::models::inference::{TokenClassifier, TokenClassifierOutput, TokenPrediction};
use crate::models::tokenization::{merge_token_spans, tokenize_with_offsets};
use crate::pipeline::report::ModuleDetail;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Configuration for hallucination detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationConfig {
    /// Confidence threshold for flagging hallucinations [0.0, 1.0]
    ///
    /// Tokens with `P(hallucinated) >= threshold` are flagged.
    pub threshold: f64,

    /// Maximum sequence length (will chunk longer sequences)
    pub max_length: usize,

    /// Chunk overlap size (tokens)
    ///
    /// When chunking long sequences, this many tokens overlap between chunks
    /// to preserve context at boundaries.
    pub chunk_overlap: usize,

    /// Whether to add special tokens (CLS, SEP)
    pub add_special_tokens: bool,

    /// Class ID representing "hallucinated" (typically 1)
    pub hallucination_class: usize,

    /// Minimum span length (characters) to report
    ///
    /// Spans shorter than this are filtered out to reduce noise.
    pub min_span_length: usize,
}

impl Default for HallucinationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 3,
        }
    }
}

/// A detected hallucination span with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationSpan {
    /// Start character position in original text
    pub start: usize,

    /// End character position in original text
    pub end: usize,

    /// The hallucinated text
    pub text: String,

    /// Average confidence score [0.0, 1.0]
    pub confidence: f64,

    /// Description of why this was flagged
    pub evidence_gap: String,
}

/// Result from hallucination detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationResult {
    /// Detected hallucinated spans
    pub hallucinated_spans: Vec<HallucinationSpan>,

    /// Aggregate hallucination score [0.0, 1.0]
    ///
    /// Computed as the fraction of tokens classified as hallucinated.
    pub aggregate_score: f64,

    /// Total tokens analyzed
    pub total_tokens: usize,

    /// Hallucinated tokens count
    pub hallucinated_tokens: usize,

    /// Inference latency (milliseconds)
    pub latency_ms: f64,
}

/// Token-level hallucination detector.
///
/// Uses a fine-tuned BERT-style model to classify each token as supported or hallucinated.
pub struct HallucinationDetector {
    /// Backend for running inference
    backend: Arc<dyn TokenClassifier>,

    /// Tokenizer for text processing
    tokenizer: Tokenizer,

    /// Detection configuration
    config: HallucinationConfig,
}

impl HallucinationDetector {
    /// Creates a new hallucination detector.
    ///
    /// # Arguments
    ///
    /// * `backend` - Model backend implementing `TokenClassifier`
    /// * `tokenizer` - `HuggingFace` tokenizer instance
    /// * `config` - Detection configuration
    pub fn new(
        backend: Arc<dyn TokenClassifier>,
        tokenizer: Tokenizer,
        config: HallucinationConfig,
    ) -> Self {
        Self {
            backend,
            tokenizer,
            config,
        }
    }

    /// Chunks text into overlapping segments that fit within `max_length`.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to chunk
    ///
    /// # Returns
    ///
    /// A vector of `(chunk_text, char_offset)` tuples where `char_offset` is the
    /// starting position of the chunk in the original text.
    fn chunk_text(&self, text: &str) -> Vec<(String, usize)> {
        // First, do a quick tokenization to see if we even need chunking
        let full_encoding = self
            .tokenizer
            .encode(text, self.config.add_special_tokens)
            .ok();

        if let Some(encoding) = full_encoding
            && encoding.get_ids().len() <= self.config.max_length
        {
            // Text fits in one chunk
            return vec![(text.to_string(), 0)];
        }

        debug!(
            "Text exceeds max_length, chunking with overlap {}",
            self.config.chunk_overlap
        );

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_chars = chars.len();

        // Effective chunk size accounting for overlap
        let chunk_size = self.config.max_length - self.config.chunk_overlap;
        let mut start_idx = 0;

        while start_idx < total_chars {
            // Calculate end index for this chunk
            let end_idx = (start_idx + self.config.max_length).min(total_chars);

            // Extract chunk
            let chunk_text: String = chars[start_idx..end_idx].iter().collect();

            chunks.push((chunk_text, start_idx));

            // Move to next chunk with overlap
            start_idx += chunk_size;

            // If we're close to the end, just include everything in the last chunk
            if start_idx + chunk_size >= total_chars && start_idx < total_chars {
                let last_chunk: String = chars[start_idx..].iter().collect();
                chunks.push((last_chunk, start_idx));
                break;
            }
        }

        debug!("Created {} chunks from {} chars", chunks.len(), total_chars);
        chunks
    }

    /// Detects hallucinations in the given text.
    ///
    /// For long texts exceeding `max_length`, automatically chunks the input
    /// and aggregates predictions across chunks using max pooling.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to analyze for hallucinations
    ///
    /// # Returns
    ///
    /// A `HallucinationResult` containing detected spans and scores.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn detect(&self, text: &str) -> Result<HallucinationResult> {
        info!("Running hallucination detection on {} chars", text.len());
        let start = std::time::Instant::now();

        // Chunk text if needed
        let chunks = self.chunk_text(text);

        if chunks.len() == 1 {
            // Single chunk - process directly
            self.detect_single_chunk(text, 0)
        } else {
            // Multiple chunks - process and aggregate
            self.detect_chunked(text, &chunks)
        }
    }

    /// Detects hallucinations in a RAG context with question and answer.
    ///
    /// This is the recommended way to use `LettuceDetect` for RAG applications.
    /// The model expects the input formatted as: `context [SEP] question [SEP] answer`.
    ///
    /// # Arguments
    ///
    /// * `context` - Supporting document(s) or context
    /// * `question` - The question asked (optional, can be empty)
    /// * `answer` - The LLM's answer to verify against the context
    ///
    /// # Returns
    ///
    /// A `HallucinationResult` containing detected spans in the answer.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let context = "France is in Europe. The capital of France is Paris.";
    /// let question = "What is the capital of France?";
    /// let answer = "The capital of France is Berlin.";
    ///
    /// let result = detector.detect_rag(context, question, answer)?;
    /// // Should detect "Berlin" as hallucinated
    /// ```
    pub fn detect_rag(
        &self,
        context: &str,
        question: &str,
        answer: &str,
    ) -> Result<HallucinationResult> {
        info!(
            "Running RAG hallucination detection: context={} chars, question={} chars, answer={} chars",
            context.len(),
            question.len(),
            answer.len()
        );

        // Format input as: context [SEP] question [SEP] answer
        // The tokenizer will add [CLS] at start and [SEP] tokens between segments
        let formatted_text = if question.is_empty() {
            format!("{context} {answer}")
        } else {
            format!("{context} {question} {answer}")
        };

        // For RAG detection, we only care about hallucinations in the answer portion
        // So we need to track where the answer starts in the formatted text
        let answer_start_char = if question.is_empty() {
            context.len() + 1 // +1 for the space
        } else {
            context.len() + question.len() + 2 // +2 for the spaces
        };

        // Run detection on formatted text
        let result = self.detect(&formatted_text)?;

        // Filter spans to only those in the answer portion
        let answer_spans: Vec<HallucinationSpan> = result
            .hallucinated_spans
            .into_iter()
            .filter_map(|mut span| {
                // Check if span overlaps with answer portion
                if span.start >= answer_start_char {
                    // Adjust span offsets to be relative to answer text
                    span.start -= answer_start_char;
                    span.end -= answer_start_char;
                    // Update text to be from answer only
                    span.text = answer
                        .get(span.start..span.end)
                        .unwrap_or(&span.text)
                        .to_string();
                    Some(span)
                } else if span.end > answer_start_char {
                    // Span starts in context/question but extends into answer
                    span.start = 0;
                    span.end -= answer_start_char;
                    span.text = answer.get(0..span.end).unwrap_or(&span.text).to_string();
                    Some(span)
                } else {
                    // Span is entirely in context/question, filter it out
                    None
                }
            })
            .collect();

        Ok(HallucinationResult {
            hallucinated_spans: answer_spans,
            aggregate_score: result.aggregate_score,
            total_tokens: result.total_tokens,
            hallucinated_tokens: result.hallucinated_tokens,
            latency_ms: result.latency_ms,
        })
    }

    /// Detects hallucinations in a single chunk (no aggregation needed).
    fn detect_single_chunk(&self, text: &str, _offset: usize) -> Result<HallucinationResult> {
        let start = std::time::Instant::now();

        // Tokenize input
        let input = tokenize_with_offsets(
            &self.tokenizer,
            text,
            self.config.add_special_tokens,
            self.config.max_length,
        )?;

        debug!("Tokenized into {} tokens", input.input_ids.len());

        // Run inference
        let outputs = self.backend.predict(std::slice::from_ref(&input))?;
        let output = &outputs[0];

        // Extract hallucinated spans
        let hallucinated_spans = self.extract_spans(&output.tokens, text);

        // Compute aggregate score (fraction of hallucinated tokens)
        let total_tokens = output.tokens.len();
        let hallucinated_tokens = output
            .tokens
            .iter()
            .filter(|t| t.predicted_class == self.config.hallucination_class)
            .count();

        #[allow(clippy::cast_precision_loss)]
        let aggregate_score = if total_tokens > 0 {
            hallucinated_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Detection complete: {} spans, {}/{} tokens hallucinated, {:.2}ms",
            hallucinated_spans.len(),
            hallucinated_tokens,
            total_tokens,
            latency_ms
        );

        Ok(HallucinationResult {
            hallucinated_spans,
            aggregate_score,
            total_tokens,
            hallucinated_tokens,
            latency_ms,
        })
    }

    /// Detects hallucinations across multiple chunks with aggregation.
    fn detect_chunked(
        &self,
        original_text: &str,
        chunks: &[(String, usize)],
    ) -> Result<HallucinationResult> {
        info!("Processing {} chunks", chunks.len());
        let start = std::time::Instant::now();

        // Process each chunk
        let mut all_predictions: Vec<(usize, usize, f64)> = Vec::new(); // (start, end, confidence)
        let mut total_tokens = 0;
        let mut hallucinated_tokens = 0;

        for (chunk_idx, (chunk_text, char_offset)) in chunks.iter().enumerate() {
            debug!("Processing chunk {}/{}", chunk_idx + 1, chunks.len());

            // Tokenize chunk
            let input = tokenize_with_offsets(
                &self.tokenizer,
                chunk_text,
                self.config.add_special_tokens,
                self.config.max_length,
            )?;

            // Run inference
            let outputs = self.backend.predict(std::slice::from_ref(&input))?;
            let output = &outputs[0];

            total_tokens += output.tokens.len();

            // Extract predictions from this chunk, adjusting offsets
            for token in &output.tokens {
                // Skip special tokens
                if token.offset.0 == token.offset.1 {
                    continue;
                }

                if token.predicted_class == self.config.hallucination_class {
                    hallucinated_tokens += 1;

                    // Adjust offsets to global text coordinates
                    let global_start = char_offset + token.offset.0;
                    let global_end = char_offset + token.offset.1;
                    let confidence = token.confidence();

                    all_predictions.push((global_start, global_end, confidence));
                }
            }
        }

        // Aggregate overlapping predictions using max pooling
        let aggregated_spans = self.aggregate_chunk_predictions(&all_predictions, original_text);

        #[allow(clippy::cast_precision_loss)]
        let aggregate_score = if total_tokens > 0 {
            hallucinated_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Detection complete: {} spans, {}/{} tokens hallucinated, {:.2}ms",
            aggregated_spans.len(),
            hallucinated_tokens,
            total_tokens,
            latency_ms
        );

        Ok(HallucinationResult {
            hallucinated_spans: aggregated_spans,
            aggregate_score,
            total_tokens,
            hallucinated_tokens,
            latency_ms,
        })
    }

    /// Aggregates predictions from multiple chunks using max pooling.
    ///
    /// For overlapping regions, takes the maximum confidence score.
    fn aggregate_chunk_predictions(
        &self,
        predictions: &[(usize, usize, f64)],
        original_text: &str,
    ) -> Vec<HallucinationSpan> {
        if predictions.is_empty() {
            return Vec::new();
        }

        // Sort predictions by start position
        let mut sorted_preds = predictions.to_vec();
        sorted_preds.sort_by_key(|(start, _, _)| *start);

        // Merge overlapping/adjacent predictions
        let mut merged: Vec<(usize, usize, f64)> = Vec::new();

        for (start, end, confidence) in sorted_preds {
            if let Some(last) = merged.last_mut() {
                // Check if this prediction overlaps or is adjacent to the last one
                if start <= last.1 {
                    // Overlapping - extend and take max confidence
                    last.1 = last.1.max(end);
                    last.2 = last.2.max(confidence);
                } else {
                    // No overlap - add as new span
                    merged.push((start, end, confidence));
                }
            } else {
                // First prediction
                merged.push((start, end, confidence));
            }
        }

        // Convert to HallucinationSpan and filter by threshold/min_length
        merged
            .into_iter()
            .filter(|(start, end, confidence)| {
                *confidence >= self.config.threshold && (end - start) >= self.config.min_span_length
            })
            .map(|(start, end, confidence)| {
                let text = original_text
                    .get(start..end)
                    .unwrap_or("<invalid offset>")
                    .to_string();

                HallucinationSpan {
                    start,
                    end,
                    text,
                    confidence,
                    evidence_gap: "No supporting evidence found in context".to_string(),
                }
            })
            .collect()
    }

    /// Extracts hallucination spans from token predictions.
    fn extract_spans(
        &self,
        tokens: &[TokenPrediction],
        original_text: &str,
    ) -> Vec<HallucinationSpan> {
        // Merge consecutive hallucinated tokens
        let raw_spans = merge_token_spans(tokens, self.config.hallucination_class);

        // Filter by confidence threshold and minimum length
        raw_spans
            .into_iter()
            .filter(|(start, end, confidence)| {
                *confidence >= self.config.threshold && (end - start) >= self.config.min_span_length
            })
            .map(|(start, end, confidence)| {
                let text = original_text
                    .get(start..end)
                    .unwrap_or("<invalid offset>")
                    .to_string();

                HallucinationSpan {
                    start,
                    end,
                    text,
                    confidence,
                    evidence_gap: "No supporting evidence found in context".to_string(),
                }
            })
            .collect()
    }

    /// Converts the result to module details for inclusion in a report.
    #[must_use]
    pub fn to_module_details(result: &HallucinationResult) -> Vec<ModuleDetail> {
        result
            .hallucinated_spans
            .iter()
            .map(|span| ModuleDetail::HallucinationSpan {
                start: span.start,
                end: span.end,
                text: span.text.clone(),
                confidence: span.confidence,
                evidence_gap: span.evidence_gap.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inference::{TokenClassifierInput, TokenClassifierOutput};
    use tokenizers::Tokenizer as TokenizerType;
    use tokenizers::models::wordpiece::WordPiece;

    /// Creates a simple mock tokenizer for testing
    fn create_mock_tokenizer() -> TokenizerType {
        // Create a simple WordPiece tokenizer (doesn't need to actually tokenize correctly for our tests)
        let wp = WordPiece::default();
        TokenizerType::new(wp)
    }

    /// Mock backend for testing
    struct MockBackend {
        /// Indices of tokens to classify as hallucinated
        hallucinated_indices: Vec<usize>,
    }

    impl TokenClassifier for MockBackend {
        fn predict(&self, inputs: &[TokenClassifierInput]) -> Result<Vec<TokenClassifierOutput>> {
            let mut outputs = Vec::new();

            for input in inputs {
                let mut tokens = Vec::new();

                for (idx, &token_id) in input.input_ids.iter().enumerate() {
                    let is_hallucinated = self.hallucinated_indices.contains(&idx);
                    let predicted_class = usize::from(is_hallucinated);
                    let probabilities = if is_hallucinated {
                        vec![0.2, 0.8]
                    } else {
                        vec![0.9, 0.1]
                    };

                    tokens.push(TokenPrediction {
                        token_id,
                        token: format!("token_{idx}"),
                        predicted_class,
                        probabilities,
                        offset: input.offsets[idx],
                    });
                }

                outputs.push(TokenClassifierOutput {
                    tokens,
                    latency_ms: 10.0,
                });
            }

            Ok(outputs)
        }

        fn device_info(&self) -> String {
            "MockDevice".to_string()
        }

        fn max_length(&self) -> usize {
            512
        }
    }

    #[test]
    fn test_hallucination_config_default() {
        let config = HallucinationConfig::default();
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.max_length, 512);
        assert_eq!(config.hallucination_class, 1);
    }

    #[test]
    fn test_extract_spans_filters_by_threshold() {
        // Two consecutive hallucinated tokens with different confidences
        // They will be merged and averaged: (0.4 + 0.8) / 2 = 0.6 >= 0.5 threshold
        let tokens = vec![
            TokenPrediction {
                token_id: 1,
                token: "low".to_string(),
                predicted_class: 1,
                probabilities: vec![0.6, 0.4], // Individual confidence 0.4
                offset: (0, 3),
            },
            TokenPrediction {
                token_id: 2,
                token: "high".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8], // Individual confidence 0.8
                offset: (4, 8),
            },
        ];

        let config = HallucinationConfig {
            threshold: 0.5,
            min_span_length: 0,
            ..Default::default()
        };

        let backend = Arc::new(MockBackend {
            hallucinated_indices: vec![],
        });
        let tokenizer = create_mock_tokenizer();
        let detector = HallucinationDetector::new(backend, tokenizer, config);

        let spans = detector.extract_spans(&tokens, "low high");

        // Tokens are merged into one span with average confidence (0.4 + 0.8) / 2 = 0.6
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "low high");
        assert!((spans[0].confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_extract_spans_filters_by_min_length() {
        // Two consecutive hallucinated tokens forming one merged span
        let tokens = vec![
            TokenPrediction {
                token_id: 1,
                token: "a".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (0, 1),
            },
            TokenPrediction {
                token_id: 2,
                token: "longer".to_string(),
                predicted_class: 1,
                probabilities: vec![0.2, 0.8],
                offset: (2, 8),
            },
        ];

        let config = HallucinationConfig {
            threshold: 0.5,
            min_span_length: 3,
            ..Default::default()
        };

        let backend = Arc::new(MockBackend {
            hallucinated_indices: vec![],
        });
        let tokenizer = create_mock_tokenizer();
        let detector = HallucinationDetector::new(backend, tokenizer, config);

        let spans = detector.extract_spans(&tokens, "a longer");

        // Merged span "a longer" has length 8 >= 3, so it's included
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "a longer");
    }

    #[test]
    fn test_to_module_details() {
        let result = HallucinationResult {
            hallucinated_spans: vec![HallucinationSpan {
                start: 0,
                end: 10,
                text: "test span".to_string(),
                confidence: 0.9,
                evidence_gap: "No evidence".to_string(),
            }],
            aggregate_score: 0.5,
            total_tokens: 10,
            hallucinated_tokens: 5,
            latency_ms: 50.0,
        };

        let details = HallucinationDetector::to_module_details(&result);

        assert_eq!(details.len(), 1);
        match &details[0] {
            ModuleDetail::HallucinationSpan {
                start,
                end,
                text,
                confidence,
                ..
            } => {
                assert_eq!(*start, 0);
                assert_eq!(*end, 10);
                assert_eq!(text, "test span");
                assert_eq!(*confidence, 0.9);
            }
            _ => panic!("Expected HallucinationSpan variant"),
        }
    }
}
