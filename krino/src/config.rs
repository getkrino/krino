//! Configuration management for Krino.
//!
//! This module handles all configuration for Krino, including model paths,
//! version pinning, performance targets, and module-specific settings.

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Global configuration for Krino evaluation engine.
///
/// This structure contains all settings required for running Krino evaluations,
/// including model configurations, performance targets, and module settings.
///
/// # Examples
///
/// ```no_run
/// use krino::config::KrinoConfig;
///
/// let config = KrinoConfig::default();
/// assert!(config.models.contains_key("modernbert-nli"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrinoConfig {
    /// Model registry: maps model IDs to their configurations
    pub models: HashMap<String, ModelConfig>,

    /// Performance targets and limits
    pub performance: PerformanceConfig,

    /// Module-specific settings
    pub modules: ModuleConfig,

    /// General settings
    pub general: GeneralConfig,
}

/// Configuration for a single model.
///
/// Each model is identified by a unique ID and includes version information,
/// file paths, and integrity verification settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Human-readable model name
    pub name: String,

    /// `HuggingFace` model ID or local path
    pub source: String,

    /// Git revision or version tag (for reproducibility)
    pub revision: Option<String>,

    /// Expected SHA256 hash of model weights (for integrity verification)
    pub sha256: Option<String>,

    /// Local cache path for model files
    pub cache_path: Option<PathBuf>,

    /// Inference backend to use (candle, onnx, etc.)
    pub backend: InferenceBackend,

    /// Whether to load model on startup or lazily on first use
    pub lazy_load: bool,
}

/// Inference backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    /// Candle-based inference (primary)
    Candle,

    /// ONNX Runtime
    Onnx,

    /// Pure Rust rule engine (no ML models)
    Rules,
}

/// Performance configuration and targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum latency per evaluation (milliseconds)
    pub max_latency_ms: u64,

    /// Maximum memory usage (megabytes)
    pub max_memory_mb: usize,

    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,

    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
}

/// Module-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModuleConfig {
    /// Hallucination detection settings
    pub hallucination: HallucinationConfig,

    /// Groundedness (NLI) settings
    pub groundedness: GroundednessConfig,

    /// Similarity evaluation settings
    pub similarity: SimilarityConfig,

    /// Schema validation settings
    pub schema: SchemaConfig,

    /// PII detection settings
    pub pii: PiiConfig,
}

/// Hallucination detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationConfig {
    /// Whether module is enabled
    pub enabled: bool,

    /// Confidence threshold for flagging hallucinations (0.0-1.0)
    pub threshold: f64,

    /// Model ID to use for hallucination detection
    pub model_id: String,
}

/// Groundedness (NLI) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundednessConfig {
    /// Whether module is enabled
    pub enabled: bool,

    /// Minimum score for passing (0.0-1.0)
    pub threshold: f64,

    /// Model ID to use for NLI
    pub model_id: String,

    /// Whether to perform claim decomposition
    pub decompose_claims: bool,

    /// Confidence threshold for flagging contradictions (0.0-1.0)
    pub contradiction_threshold: f64,

    /// Whether NEUTRAL verdicts count as unsupported (stricter mode)
    pub treat_neutral_as_unsupported: bool,

    /// Maximum context chunk size in tokens (0 = no chunking)
    pub max_context_tokens: usize,

    /// Overlap between context chunks in tokens
    pub chunk_overlap_tokens: usize,

    /// Minimum claim length in characters to evaluate
    pub min_claim_length: usize,
}

/// Similarity evaluation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Whether module is enabled
    pub enabled: bool,

    /// Minimum similarity score for passing (0.0-1.0)
    pub threshold: f64,

    /// Model ID to use for embeddings
    pub model_id: String,
}

/// Schema validation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    /// Whether module is enabled
    pub enabled: bool,

    /// Whether to validate JSON structure
    pub validate_json: bool,

    /// Whether to check type constraints
    pub validate_types: bool,
}

/// PII detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiConfig {
    /// Whether module is enabled
    pub enabled: bool,

    /// Types of PII to detect
    pub entity_types: Vec<String>,

    /// Model ID for NER (if using ML-based detection)
    pub model_id: Option<String>,

    /// Whether to use regex patterns
    pub use_patterns: bool,
}

/// General configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Logging level (trace, debug, info, warn, error)
    pub log_level: String,

    /// Whether to enable strict determinism checks
    pub strict_determinism: bool,

    /// Output format for reports (json, yaml, etc.)
    pub output_format: String,
}

impl Default for KrinoConfig {
    fn default() -> Self {
        Self {
            models: Self::default_models(),
            performance: PerformanceConfig::default(),
            modules: ModuleConfig::default(),
            general: GeneralConfig::default(),
        }
    }
}

impl KrinoConfig {
    /// Creates default model configurations.
    ///
    /// These are placeholder configurations that will be updated when
    /// actual models are integrated.
    fn default_models() -> HashMap<String, ModelConfig> {
        let mut models = HashMap::new();

        // Placeholder for ModernBERT-based hallucination detector
        models.insert(
            "modernbert-hallucination".to_string(),
            ModelConfig {
                name: "ModernBERT Hallucination Detector".to_string(),
                source: "answerdotai/ModernBERT-base".to_string(),
                revision: None,
                sha256: None,
                cache_path: None,
                backend: InferenceBackend::Candle,
                lazy_load: true,
            },
        );

        // Placeholder for DeBERTa-MNLI
        models.insert(
            "deberta-mnli".to_string(),
            ModelConfig {
                name: "DeBERTa-v3 MNLI".to_string(),
                source: "microsoft/deberta-v3-base".to_string(),
                revision: None,
                sha256: None,
                cache_path: None,
                backend: InferenceBackend::Onnx,
                lazy_load: true,
            },
        );

        // DeBERTa-v3-large NLI for groundedness checking (quantized by default for performance)
        models.insert(
            "deberta-nli".to_string(),
            ModelConfig {
                name: "DeBERTa-v3-large NLI (MNLI+FEVER+ANLI+LingNLI+WANLI) - Quantized"
                    .to_string(),
                source: "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli".to_string(),
                revision: None,
                sha256: Some(
                    "8624f8205fdfb38d551e505e870d7a565ddd414c7834f416fbc2907df231510b".to_string(),
                ),
                cache_path: Some(PathBuf::from("models/deberta-nli-onnx-quantized")),
                backend: InferenceBackend::Onnx,
                lazy_load: true,
            },
        );

        models
    }

    /// Loads configuration from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::NotFound` if the file doesn't exist.
    /// Returns `ConfigError::ParseFailed` if the file is invalid JSON.
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let contents = std::fs::read_to_string(&path)
            .map_err(|_| ConfigError::NotFound { path: path.clone() })?;

        serde_json::from_str(&contents).map_err(|e| {
            ConfigError::ParseFailed {
                path,
                reason: e.to_string(),
            }
            .into()
        })
    }

    /// Saves configuration to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialization fails.
    pub fn to_file(&self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validates the configuration.
    ///
    /// Checks that all required fields are present and values are within
    /// acceptable ranges.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::InvalidValue` if any setting is invalid.
    pub fn validate(&self) -> Result<()> {
        // Validate performance settings
        if self.performance.max_latency_ms == 0 {
            return Err(ConfigError::invalid_value(
                "performance.max_latency_ms",
                "must be greater than 0",
            )
            .into());
        }

        // Validate module thresholds
        Self::validate_threshold(
            "hallucination.threshold",
            self.modules.hallucination.threshold,
        )?;
        Self::validate_threshold(
            "groundedness.threshold",
            self.modules.groundedness.threshold,
        )?;
        Self::validate_threshold("similarity.threshold", self.modules.similarity.threshold)?;

        Ok(())
    }

    /// Validates that a threshold is in the range [0.0, 1.0].
    fn validate_threshold(field: &str, threshold: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ConfigError::invalid_value(field, "must be between 0.0 and 1.0").into());
        }
        Ok(())
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 200,
            max_memory_mb: 2048,
            enable_monitoring: true,
            num_threads: None, // Use default thread pool size
        }
    }
}

impl Default for HallucinationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.7,
            model_id: "modernbert-hallucination".to_string(),
        }
    }
}

impl Default for GroundednessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.5,
            model_id: "deberta-mnli".to_string(),
            decompose_claims: true,
            contradiction_threshold: 0.7,
            treat_neutral_as_unsupported: false,
            max_context_tokens: 900, // Leave room for claim + special tokens in 1024 window
            chunk_overlap_tokens: 100,
            min_claim_length: 10,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.7,
            model_id: "sentence-transformer".to_string(),
        }
    }
}

impl Default for SchemaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            validate_json: true,
            validate_types: true,
        }
    }
}

impl Default for PiiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            entity_types: vec![
                "email".to_string(),
                "phone".to_string(),
                "ssn".to_string(),
                "credit_card".to_string(),
            ],
            model_id: None,
            use_patterns: true,
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            strict_determinism: true,
            output_format: "json".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KrinoConfig::default();
        assert!(config.models.contains_key("modernbert-hallucination"));
        assert!(config.models.contains_key("deberta-mnli"));
        assert_eq!(config.performance.max_latency_ms, 200);
    }

    #[test]
    fn test_config_validation_success() {
        let config = KrinoConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_threshold() {
        let mut config = KrinoConfig::default();
        config.modules.hallucination.threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_latency() {
        let mut config = KrinoConfig::default();
        config.performance.max_latency_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_inference_backend_serialization() {
        let backend = InferenceBackend::Candle;
        let json = serde_json::to_string(&backend).unwrap();
        assert_eq!(json, "\"candle\"");
    }

    #[test]
    fn test_model_config_creation() {
        let model = ModelConfig {
            name: "Test Model".to_string(),
            source: "test/model".to_string(),
            revision: Some("abc123".to_string()),
            sha256: Some("def456".to_string()),
            cache_path: Some(PathBuf::from("/tmp/models")),
            backend: InferenceBackend::Candle,
            lazy_load: false,
        };

        assert_eq!(model.name, "Test Model");
        assert_eq!(model.backend, InferenceBackend::Candle);
        assert!(!model.lazy_load);
    }
}
