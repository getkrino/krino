//! Krino: Deterministic LLM Evaluation Engine
//!
//! Copyright (c) 2024-2025 Justin Smith. All Rights Reserved.
//! This software is proprietary and confidential.
//! Unauthorized copying, distribution, or use is strictly prohibited.
//!
//! Krino is a deterministic evaluation engine for Large Language Model (LLM) outputs.
//! It uses purpose-built NLI models, token-level classifiers, and statistical tests —
//! never using another LLM as a judge.
//!
//! # Core Value Proposition
//!
//! **Same inputs. Same results. Every time.**
//!
//! All evaluations are deterministic, explainable, and auditable. This makes Krino
//! suitable for CI/CD pipelines, compliance audits, and reproducible research.
//!
//! # Foundational Principles
//!
//! 1. **Determinism is Non-Negotiable**: Every evaluation produces identical results
//!    given identical inputs. If a result cannot be reproduced, it is a bug.
//!
//! 2. **No LLM-as-Judge**: Krino never calls a generative LLM to evaluate another
//!    LLM's output. All evaluation uses specialized discriminative models or rules.
//!
//! 3. **Explain, Don't Just Score**: Every evaluation includes specific evidence,
//!    token spans, and human-readable explanations.
//!
//! 4. **Context Required for Factual Verification**: Krino performs textual entailment
//!    against provided context, not truth-checking against world knowledge.
//!
//! 5. **Speed is a Feature**: Target latency is <200ms per evaluation.
//!
//! 6. **Privacy by Default**: All evaluation runs locally. No data leaves the
//!    user's environment.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use krino::{KrinoConfig, KrinoEngine};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create engine with default configuration
//! let config = KrinoConfig::default();
//! let engine = KrinoEngine::new(config)?;
//!
//! // Evaluate LLM output (placeholder - full implementation coming in Phase 2)
//! // let report = engine.evaluate(
//! //     "The company was founded in 1994.",  // context
//! //     "When was the company founded?",     // question
//! //     "It was founded in 1987."            // answer
//! // )?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! Krino consists of several key components:
//!
//! - **Models**: Model loading and inference backends (Candle, ONNX, rules)
//! - **Modules**: Evaluation modules (hallucination, groundedness, similarity, etc.)
//! - **Pipeline**: Evaluation orchestration and report generation
//! - **Config**: Configuration management
//! - **Error**: Comprehensive error handling
//!
//! # Feature Flags
//!
//! - `cli`: Enables command-line interface (disabled by default)

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)] // TODO: Remove once docs are complete
#![allow(clippy::module_name_repetitions)]
#![allow(missing_docs)] // TODO: Remove once all struct fields are documented
#![allow(unused)] // TODO: Remove once Phase 1 is complete and everything is wired together
#![allow(dead_code)] // TODO: Remove once Phase 1 is complete
#![allow(clippy::float_cmp)] // Allow exact float comparisons in tests (deterministic values)

// Public modules
pub mod config;
pub mod error;
pub mod models;
pub mod modules;
pub mod pipeline;

// Re-exports for convenience
pub use config::KrinoConfig;
pub use error::{KrinoError, Result};

use tracing::info;

/// Main entry point for Krino evaluation engine.
///
/// The `KrinoEngine` manages model loading, evaluation orchestration, and report
/// generation. It is the primary interface for all evaluation operations.
///
/// # Examples
///
/// ```rust,no_run
/// use krino::{KrinoConfig, KrinoEngine};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = KrinoConfig::default();
/// let engine = KrinoEngine::new(config)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct KrinoEngine {
    config: KrinoConfig,
    // Model registry will be added in next phase
    // models: ModelRegistry,
}

impl KrinoEngine {
    /// Creates a new Krino engine with the given configuration.
    ///
    /// This validates the configuration and prepares the engine for evaluation.
    /// Models are loaded lazily on first use (unless configured otherwise).
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use krino::{KrinoConfig, KrinoEngine};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = KrinoConfig::default();
    /// let engine = KrinoEngine::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: KrinoConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        info!(
            "Krino engine initialized with {} models configured",
            config.models.len()
        );

        Ok(Self { config })
    }

    /// Returns a reference to the engine's configuration.
    #[must_use]
    pub fn config(&self) -> &KrinoConfig {
        &self.config
    }

    /// Returns the version of Krino.
    ///
    /// This is included in all evaluation reports for reproducibility tracking.
    #[must_use]
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

/// Initializes the Krino tracing subscriber for logging.
///
/// This sets up structured logging using the `tracing` crate. Call this at the
/// start of your application to enable logging.
///
/// # Examples
///
/// ```rust,no_run
/// krino::init_tracing();
/// ```
pub fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = KrinoConfig::default();
        let engine = KrinoEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_config_access() {
        let config = KrinoConfig::default();
        let engine = KrinoEngine::new(config).unwrap();
        assert_eq!(engine.config().performance.max_latency_ms, 200);
    }

    #[test]
    fn test_version() {
        let version = KrinoEngine::version();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_invalid_config_rejected() {
        let mut config = KrinoConfig::default();
        config.performance.max_latency_ms = 0; // Invalid
        let engine = KrinoEngine::new(config);
        assert!(engine.is_err());
    }
}
