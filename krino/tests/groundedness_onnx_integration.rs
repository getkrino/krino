//! Integration test for groundedness checking with real ONNX NLI backend.
//!
//! This test validates that the groundedness module works correctly with
//! the production ONNX DeBERTa model for real-world RAG scenarios.
//!
//! Run with: cargo test --test groundedness_onnx_integration -- --ignored

use krino::error::Result;
use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::EmbeddingSimilarity;
use krino::modules::groundedness::{GroundednessChecker, GroundednessConfig};
use std::path::Path;
use std::sync::Arc;

/// Mock embedding backend for testing (pre-filtering disabled in tests)
struct MockEmbedding;

impl EmbeddingSimilarity for MockEmbedding {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Simple character frequency embedding for testing
        Ok(texts
            .iter()
            .map(|text| {
                let mut vec = vec![0.0_f32; 26];
                for ch in text.to_lowercase().chars() {
                    if ch.is_ascii_lowercase() {
                        vec[(ch as u8 - b'a') as usize] += 1.0;
                    }
                }
                // Normalize
                let sum: f32 = vec.iter().sum();
                if sum > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= sum);
                }
                vec
            })
            .collect())
    }

    fn embedding_dim(&self) -> usize {
        26 // Character frequency vector size
    }

    fn device_info(&self) -> String {
        "MockEmbedding".to_string()
    }
}

#[test]
#[ignore]
fn test_groundedness_with_onnx_entailment() {
    let model_path = Path::new("models/deberta-nli-onnx");

    // Check if model exists
    if !model_path.join("model.onnx").exists() {
        eprintln!("❌ Model not found at {}", model_path.display());
        eprintln!("   Run: cd scripts && uv run export_deberta_onnx.py");
        panic!("Model files not found");
    }

    // Load real ONNX NLI model
    let nli_backend = Arc::new(
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model"),
    );

    let embedding_backend = Arc::new(MockEmbedding);

    // Disable pre-filtering for deterministic testing
    let config = GroundednessConfig {
        top_k_context: 0, // Evaluate all context sentences
        include_entailment_matrix: true,
        ..Default::default()
    };

    let checker = GroundednessChecker::new(nli_backend, embedding_backend, config);

    // Test case 1: Clear entailment
    println!("\n📋 Test 1: Clear Entailment");
    let context = "Paris is the capital and largest city of France.";
    let output = "The capital of France is Paris.";

    let result = checker.check(context, output).expect("Check failed");

    println!("  Context: {context}");
    println!("  Output: {output}");
    println!("  Faithfulness score: {:.3}", result.faithfulness_score);
    println!("  Total claims: {}", result.total_claims);
    println!("  Supported claims: {}", result.supported_claims);

    // Should have high faithfulness (claim is entailed by context)
    assert!(
        result.faithfulness_score >= 0.9,
        "Expected high faithfulness for entailed claim, got {:.3}",
        result.faithfulness_score
    );
    assert_eq!(result.supported_claims, 1);
    assert_eq!(result.contradicted_claims, 0);
}

#[test]
#[ignore]
fn test_groundedness_with_onnx_contradiction() {
    let model_path = Path::new("models/deberta-nli-onnx");

    if !model_path.join("model.onnx").exists() {
        panic!("Model files not found");
    }

    let nli_backend = Arc::new(
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model"),
    );

    let embedding_backend = Arc::new(MockEmbedding);

    let config = GroundednessConfig {
        top_k_context: 0,
        include_entailment_matrix: true,
        contradiction_threshold: 0.7,
        ..Default::default()
    };

    let checker = GroundednessChecker::new(nli_backend, embedding_backend, config);

    // Test case 2: Clear contradiction
    println!("\n📋 Test 2: Clear Contradiction");
    let context = "The store is open every day of the week.";
    let output = "The store is closed on Sundays.";

    let result = checker.check(context, output).expect("Check failed");

    println!("  Context: {context}");
    println!("  Output: {output}");
    println!("  Faithfulness score: {:.3}", result.faithfulness_score);
    println!("  Contradicted claims: {}", result.contradicted_claims);

    // Should detect contradiction
    assert!(
        result.contradicted_claims >= 1,
        "Expected contradiction to be detected"
    );
    assert!(
        result.faithfulness_score < 0.5,
        "Expected low faithfulness for contradicted claim, got {:.3}",
        result.faithfulness_score
    );
}

#[test]
#[ignore]
fn test_groundedness_with_onnx_rag_scenario() {
    let model_path = Path::new("models/deberta-nli-onnx");

    if !model_path.join("model.onnx").exists() {
        panic!("Model files not found");
    }

    let nli_backend = Arc::new(
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model"),
    );

    let embedding_backend = Arc::new(MockEmbedding);

    let config = GroundednessConfig {
        top_k_context: 0,
        include_entailment_matrix: true,
        ..Default::default()
    };

    let checker = GroundednessChecker::new(nli_backend, embedding_backend, config);

    // Test case 3: Realistic RAG scenario with multiple claims
    println!("\n📋 Test 3: RAG Scenario (Multiple Claims)");
    let context = "The DeBERTa-v3-large model was trained on MNLI, FEVER, ANLI, and other NLI datasets. \
                   It achieves state-of-the-art performance on natural language inference tasks. \
                   The model uses disentangled attention mechanisms.";

    let output = "DeBERTa-v3-large was trained on multiple NLI datasets including MNLI and FEVER. \
                  It uses disentangled attention.";

    let result = checker.check(context, output).expect("Check failed");

    println!("  Context: {} chars", context.len());
    println!("  Output: {} chars", output.len());
    println!("  Faithfulness score: {:.3}", result.faithfulness_score);
    println!("  Total claims: {}", result.total_claims);
    println!(
        "  Supported: {}, Neutral: {}, Contradicted: {}",
        result.supported_claims, result.neutral_claims, result.contradicted_claims
    );

    // Print per-claim verdicts
    for (i, verdict) in result.verdicts.iter().enumerate() {
        println!("\n  Claim {}: \"{}\"", i + 1, verdict.claim);
        println!("    Label: {}", verdict.label);
        println!("    Supported: {}", verdict.supported);
        println!("    Entailment prob: {:.3}", verdict.entailment_prob);
        if let Some(evidence) = &verdict.best_evidence {
            println!("    Best evidence: \"{}\"", evidence.sentence);
        }
    }

    // Both claims should be entailed
    assert!(
        result.faithfulness_score >= 0.8,
        "Expected high faithfulness for factual RAG output, got {:.3}",
        result.faithfulness_score
    );
}

#[test]
#[ignore]
fn test_groundedness_performance_baseline() {
    let model_path = Path::new("models/deberta-nli-onnx");

    if !model_path.join("model.onnx").exists() {
        panic!("Model files not found");
    }

    let nli_backend = Arc::new(
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model"),
    );

    let embedding_backend = Arc::new(MockEmbedding);

    let config = GroundednessConfig {
        top_k_context: 0,
        ..Default::default()
    };

    let checker = GroundednessChecker::new(nli_backend, embedding_backend, config);

    // Performance test: measure end-to-end latency
    println!("\n⚡ Performance Baseline");
    let context = "Natural language inference is the task of determining whether a hypothesis is true, false, or undetermined given a premise.";
    let output = "NLI determines if a hypothesis follows from a premise.";

    let start = std::time::Instant::now();
    let result = checker.check(context, output).expect("Check failed");
    let latency = start.elapsed();

    println!("  Context sentences: {}", result.context_sentences.len());
    println!("  Output claims: {}", result.output_claims.len());
    println!(
        "  Total NLI calls: {} (context × claims)",
        result.context_sentences.len() * result.output_claims.len()
    );
    println!(
        "  End-to-end latency: {:.1}ms",
        latency.as_secs_f64() * 1000.0
    );
    println!("  Faithfulness score: {:.3}", result.faithfulness_score);

    // Target: <200ms warm; allow up to 1000ms to accommodate cold ONNX session start
    assert!(
        latency.as_millis() < 1000,
        "Expected latency <1000ms, got {}ms (this is for 1 context × 1 claim)",
        latency.as_millis()
    );
}
