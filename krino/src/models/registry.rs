//! Model registry for version management and integrity verification.
//!
//! This module manages model loading, version tracking, and SHA256-based
//! integrity verification to ensure reproducibility across environments.

use crate::config::{InferenceBackend, ModelConfig};
use crate::error::{ModelError, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Registry for managing model versions and loaded models.
///
/// The registry tracks all available models, their configurations, and provides
/// methods for loading and verifying model integrity.
///
/// # Determinism
///
/// All models are versioned using SHA256 hashes. Evaluation results include
/// model hashes to ensure reproducibility across different runs and environments.
#[derive(Debug)]
pub struct ModelRegistry {
    /// Map of model ID to model configuration
    configs: HashMap<String, ModelConfig>,

    /// Map of model ID to verified SHA256 hash
    verified_hashes: HashMap<String, String>,

    /// Base directory for model cache
    cache_dir: PathBuf,
}

impl ModelRegistry {
    /// Creates a new model registry.
    ///
    /// # Arguments
    ///
    /// * `configs` - Map of model IDs to their configurations
    /// * `cache_dir` - Base directory for caching model files
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use krino::models::ModelRegistry;
    /// use std::collections::HashMap;
    /// use std::path::PathBuf;
    ///
    /// let configs = HashMap::new();
    /// let registry = ModelRegistry::new(configs, PathBuf::from("/tmp/models"));
    /// ```
    pub fn new(configs: HashMap<String, ModelConfig>, cache_dir: PathBuf) -> Self {
        info!(
            "Initializing model registry with {} models, cache_dir: {}",
            configs.len(),
            cache_dir.display()
        );

        Self {
            configs,
            verified_hashes: HashMap::new(),
            cache_dir,
        }
    }

    /// Gets the configuration for a model.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::NotInitialized` if the model ID is not registered.
    pub fn get_config(&self, model_id: &str) -> Result<&ModelConfig> {
        self.configs.get(model_id).ok_or_else(|| {
            ModelError::NotInitialized {
                model_id: model_id.to_string(),
            }
            .into()
        })
    }

    /// Gets the verified SHA256 hash for a model.
    ///
    /// Returns `None` if the model hasn't been verified yet.
    pub fn get_verified_hash(&self, model_id: &str) -> Option<&str> {
        self.verified_hashes.get(model_id).map(String::as_str)
    }

    /// Lists all registered model IDs.
    #[must_use]
    pub fn list_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    /// Computes SHA256 hash of a file.
    ///
    /// This is used for integrity verification of model weights.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn compute_file_hash(path: &Path) -> Result<String> {
        debug!("Computing SHA256 hash for: {}", path.display());

        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let hash = format!("{:x}", hasher.finalize());
        debug!("Computed hash: {}", hash);

        Ok(hash)
    }

    /// Verifies the integrity of a model file using SHA256.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier
    /// * `path` - Path to the model file
    ///
    /// # Errors
    ///
    /// Returns `ModelError::IntegrityCheckFailed` if the hash doesn't match.
    /// Returns `ModelError::NotFound` if the file doesn't exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use krino::models::ModelRegistry;
    /// use std::collections::HashMap;
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let configs = HashMap::new();
    /// let mut registry = ModelRegistry::new(configs, PathBuf::from("/tmp"));
    /// // registry.verify_integrity("my-model", &PathBuf::from("model.bin"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_integrity(&mut self, model_id: &str, path: &Path) -> Result<()> {
        let config = self.get_config(model_id)?;

        // If no expected hash is configured, skip verification but warn
        let Some(expected_hash) = &config.sha256 else {
            warn!(
                "No SHA256 hash configured for model '{}', skipping integrity check",
                model_id
            );
            return Ok(());
        };

        if !path.exists() {
            return Err(ModelError::not_found(path).into());
        }

        let actual_hash = Self::compute_file_hash(path)?;

        if &actual_hash != expected_hash {
            return Err(ModelError::IntegrityCheckFailed {
                expected: expected_hash.clone(),
                actual: actual_hash,
            }
            .into());
        }

        info!(
            "Integrity verification passed for model '{}': {}",
            model_id, actual_hash
        );

        // Store verified hash
        self.verified_hashes
            .insert(model_id.to_string(), actual_hash);

        Ok(())
    }

    /// Resolves the full path to a model file.
    ///
    /// If the model configuration specifies a cache path, it is used directly.
    /// Otherwise, the path is constructed from the cache directory and model ID.
    pub fn resolve_model_path(&self, model_id: &str) -> Result<PathBuf> {
        let config = self.get_config(model_id)?;

        if let Some(cache_path) = &config.cache_path {
            return Ok(cache_path.clone());
        }

        // Default: cache_dir/model_id/model.bin
        let path = self.cache_dir.join(model_id).join("model.bin");

        Ok(path)
    }

    /// Checks if a model is available (file exists).
    #[must_use]
    pub fn is_model_available(&self, model_id: &str) -> bool {
        if let Ok(path) = self.resolve_model_path(model_id) {
            path.exists()
        } else {
            false
        }
    }

    /// Gets the inference backend for a model.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::NotInitialized` if the model ID is not registered.
    pub fn get_backend(&self, model_id: &str) -> Result<InferenceBackend> {
        Ok(self.get_config(model_id)?.backend)
    }

    /// Gets all verified model hashes as a map (for inclusion in reports).
    ///
    /// This is used in `KrinoReport` to track which model versions were used
    /// for reproducibility.
    #[must_use]
    pub fn get_all_verified_hashes(&self) -> HashMap<String, String> {
        self.verified_hashes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InferenceBackend;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_model_config(backend: InferenceBackend) -> ModelConfig {
        ModelConfig {
            name: "Test Model".to_string(),
            source: "test/model".to_string(),
            revision: None,
            sha256: None,
            cache_path: None,
            backend,
            lazy_load: true,
        }
    }

    #[test]
    fn test_registry_creation() {
        let mut configs = HashMap::new();
        configs.insert(
            "test-model".to_string(),
            create_test_model_config(InferenceBackend::Candle),
        );

        let registry = ModelRegistry::new(configs, PathBuf::from("/tmp"));
        assert_eq!(registry.list_models().len(), 1);
    }

    #[test]
    fn test_get_config() {
        let mut configs = HashMap::new();
        configs.insert(
            "test-model".to_string(),
            create_test_model_config(InferenceBackend::Candle),
        );

        let registry = ModelRegistry::new(configs, PathBuf::from("/tmp"));
        let config = registry.get_config("test-model");
        assert!(config.is_ok());
        assert_eq!(config.unwrap().name, "Test Model");
    }

    #[test]
    fn test_get_config_not_found() {
        let registry = ModelRegistry::new(HashMap::new(), PathBuf::from("/tmp"));
        let config = registry.get_config("nonexistent");
        assert!(config.is_err());
    }

    #[test]
    fn test_compute_file_hash() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Write specific content
        std::fs::write(&file_path, b"test content").unwrap();

        let hash1 = ModelRegistry::compute_file_hash(&file_path).unwrap();

        // Verify determinism: same file should produce same hash
        let hash2 = ModelRegistry::compute_file_hash(&file_path).unwrap();
        assert_eq!(hash1, hash2, "Hash should be deterministic");

        // Verify hash changes with different content
        std::fs::write(&file_path, b"different content").unwrap();
        let hash3 = ModelRegistry::compute_file_hash(&file_path).unwrap();
        assert_ne!(
            hash1, hash3,
            "Different content should produce different hash"
        );
    }

    #[test]
    fn test_verify_integrity_success() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("model.bin");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "test content").unwrap();

        let expected_hash = ModelRegistry::compute_file_hash(&file_path).unwrap();

        let mut configs = HashMap::new();
        let mut config = create_test_model_config(InferenceBackend::Candle);
        config.sha256 = Some(expected_hash.clone());
        config.cache_path = Some(file_path.clone());
        configs.insert("test-model".to_string(), config);

        let mut registry = ModelRegistry::new(configs, temp_dir.path().to_path_buf());
        let result = registry.verify_integrity("test-model", &file_path);
        assert!(result.is_ok());

        let verified = registry.get_verified_hash("test-model");
        assert_eq!(verified, Some(expected_hash.as_str()));
    }

    #[test]
    fn test_verify_integrity_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("model.bin");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "test content").unwrap();

        let mut configs = HashMap::new();
        let mut config = create_test_model_config(InferenceBackend::Candle);
        config.sha256 = Some("incorrect_hash".to_string());
        config.cache_path = Some(file_path.clone());
        configs.insert("test-model".to_string(), config);

        let mut registry = ModelRegistry::new(configs, temp_dir.path().to_path_buf());
        let result = registry.verify_integrity("test-model", &file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_backend() {
        let mut configs = HashMap::new();
        configs.insert(
            "candle-model".to_string(),
            create_test_model_config(InferenceBackend::Candle),
        );
        configs.insert(
            "onnx-model".to_string(),
            create_test_model_config(InferenceBackend::Onnx),
        );

        let registry = ModelRegistry::new(configs, PathBuf::from("/tmp"));
        assert_eq!(
            registry.get_backend("candle-model").unwrap(),
            InferenceBackend::Candle
        );
        assert_eq!(
            registry.get_backend("onnx-model").unwrap(),
            InferenceBackend::Onnx
        );
    }
}
