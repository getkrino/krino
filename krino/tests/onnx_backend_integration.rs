//! Integration test for ONNX backend validation against Python reference outputs.
//!
//! This test validates that the Rust ONNX backend produces identical results to
//! the Python ONNX Runtime implementation.
//!
//! Run with: cargo test --test onnx_backend_integration -- --ignored

use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::{SequenceClassifier, SequenceClassifierInput};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ReferenceOutput {
    premise: String,
    hypothesis: String,
    expected_class: String,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
    predicted_class: usize,
}

#[test]
#[ignore]
fn test_onnx_deberta_matches_reference() {
    let model_path = Path::new("models/deberta-nli-onnx");

    // Check if model exists
    if !model_path.join("model.onnx").exists() {
        eprintln!("❌ Model not found at {}", model_path.display());
        eprintln!("   Run: cd scripts && uv run export_deberta_onnx.py");
        panic!("Model files not found");
    }

    // Load ONNX model
    let classifier =
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model");

    // Load reference outputs
    let reference: Vec<ReferenceOutput> = serde_json::from_str(
        &std::fs::read_to_string("tests/fixtures/deberta_reference_outputs.json")
            .expect("Reference file missing"),
    )
    .expect("Invalid reference JSON");

    println!(
        "\n🧪 Validating {} test cases against Python reference...",
        reference.len()
    );

    for (i, ref_output) in reference.iter().enumerate() {
        println!("\nCase {}: {}", i + 1, ref_output.expected_class);
        println!("  Premise: {}", ref_output.premise);
        println!("  Hypothesis: {}", ref_output.hypothesis);

        let input = SequenceClassifierInput {
            text_a: ref_output.premise.clone(),
            text_b: ref_output.hypothesis.clone(),
        };
        let result = classifier.classify(&[input]).expect("Inference failed");

        let output = &result[0];

        // Critical: predicted class must match exactly
        assert_eq!(
            output.predicted_class, ref_output.predicted_class,
            "Case {i}: predicted class mismatch: Rust={}, Python={}",
            output.predicted_class, ref_output.predicted_class
        );

        // Probability tolerance: f32 ONNX inference may differ slightly from f64 numpy
        for (j, (rust_p, py_p)) in output
            .probabilities
            .iter()
            .zip(ref_output.probabilities.iter())
            .enumerate()
        {
            let diff = (rust_p - py_p).abs();
            assert!(
                diff < 1e-4,
                "Case {i}, label {j}: probability mismatch: rust={rust_p}, python={py_p}, diff={diff}"
            );
        }

        println!(
            "  ✓ Rust prediction: {} ({:.6})",
            output.predicted_label, output.probabilities[output.predicted_class]
        );
        println!(
            "    Python prediction: class {} ({:.6})",
            ref_output.predicted_class, ref_output.probabilities[ref_output.predicted_class]
        );
        println!(
            "    Max probability diff: {:.2e}",
            output
                .probabilities
                .iter()
                .zip(ref_output.probabilities.iter())
                .map(|(r, p)| (r - p).abs())
                .fold(0.0, f64::max)
        );
    }

    println!(
        "\n✅ All {} test cases match Python reference!",
        reference.len()
    );
}
