//! Model loading and inference infrastructure.
//!
//! This module provides abstractions for loading ML models and running inference
//! using different backends (Candle, ONNX, etc.).

pub mod backends;
pub mod inference;
pub mod registry;
pub mod tokenization;

// Re-exports
pub use inference::{
    PaddingStrategy, TokenClassifier, TokenClassifierInput, TokenClassifierOutput, TokenPrediction,
    TokenizerConfig, TruncationStrategy,
};
pub use registry::ModelRegistry;
