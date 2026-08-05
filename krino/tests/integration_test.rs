//! Integration tests for Krino.
//!
//! These tests verify that the major components work together correctly.

use krino::{KrinoConfig, KrinoEngine};

#[test]
fn test_engine_creation_with_default_config() {
    let config = KrinoConfig::default();
    let engine = KrinoEngine::new(config);
    assert!(engine.is_ok(), "Engine should be created successfully");
}

#[test]
fn test_engine_rejects_invalid_config() {
    let mut config = KrinoConfig::default();
    config.performance.max_latency_ms = 0; // Invalid
    let engine = KrinoEngine::new(config);
    assert!(
        engine.is_err(),
        "Engine should reject invalid configuration"
    );
}

#[test]
fn test_config_roundtrip() {
    let config = KrinoConfig::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    let deserialized: KrinoConfig = serde_json::from_str(&json).unwrap();

    // Verify key fields match
    assert_eq!(
        config.performance.max_latency_ms,
        deserialized.performance.max_latency_ms
    );
    assert_eq!(
        config.modules.hallucination.threshold,
        deserialized.modules.hallucination.threshold
    );
}

#[test]
fn test_version_is_semver() {
    let version = KrinoEngine::version();
    assert!(!version.is_empty(), "Version should not be empty");
    assert!(
        version.chars().any(|c| c == '.'),
        "Version should contain dots (semver format)"
    );
}

#[test]
fn test_config_validation() {
    let config = KrinoConfig::default();
    assert!(config.validate().is_ok(), "Default config should be valid");
}

#[test]
fn test_determinism_config_enabled_by_default() {
    let config = KrinoConfig::default();
    assert!(
        config.general.strict_determinism,
        "Strict determinism should be enabled by default"
    );
}
