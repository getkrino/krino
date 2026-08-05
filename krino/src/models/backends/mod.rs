//! Inference backend implementations.
//!
//! This module contains concrete implementations of the inference traits
//! for different ML frameworks.

pub mod candle;
pub mod modernbert_seq;
pub mod onnx;

// Re-exports
pub use candle::CandleBackend;
pub use modernbert_seq::ModernBertSeqBackend;
pub use onnx::OnnxSequenceClassifier;
