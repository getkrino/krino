// Smoke test for ONNX backend with real DeBERTa model
//
// This test is marked #[ignore] because it requires the exported model files.
// Run with: cargo test --test onnx_backend_smoke_test -- --ignored

use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::{SequenceClassifier, SequenceClassifierInput};
use std::path::Path;

#[test]
#[ignore]
fn test_onnx_deberta_loads_and_infers() {
    let model_path = Path::new("models/deberta-nli-onnx");

    // Check if model exists
    if !model_path.join("model.onnx").exists() {
        eprintln!("❌ Model not found at {}", model_path.display());
        eprintln!("   Run: cd scripts && uv run export_deberta_onnx.py");
        panic!("Model files not found");
    }

    // Load the model
    println!("Loading ONNX model from {}...", model_path.display());
    let classifier =
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model");

    println!("✓ Model loaded successfully");
    println!("  Device: {}", classifier.device_info());
    println!("  Max length: {}", classifier.max_length());
    println!("  Labels: {:?}", classifier.label_map());

    // Test inference with clear entailment case
    let test_cases = [
        (
            "The cat sat on the mat.",
            "An animal was on the mat.",
            "entailment",
        ),
        (
            "The store is open every day.",
            "The store is closed on Sundays.",
            "contradiction",
        ),
        (
            "A man is walking in the park.",
            "The man is wearing a hat.",
            "neutral",
        ),
    ];

    println!("\n🧪 Running {} test cases...", test_cases.len());

    for (i, (premise, hypothesis, expected)) in test_cases.iter().enumerate() {
        println!("\nCase {}: {expected}", i + 1);
        println!("  Premise: {premise}");
        println!("  Hypothesis: {hypothesis}");

        let input = SequenceClassifierInput {
            text_a: premise.to_string(),
            text_b: hypothesis.to_string(),
        };

        let result = classifier.classify(&[input]).expect("Inference failed");

        let output = &result[0];

        println!(
            "  → Predicted: {} ({:.3})",
            output.predicted_label, output.probabilities[output.predicted_class]
        );
        println!(
            "    Probabilities: [ent={:.3}, neu={:.3}, con={:.3}]",
            output.probabilities[0], output.probabilities[1], output.probabilities[2]
        );
        println!("    Latency: {:.1}ms", output.latency_ms);

        // Check prediction matches expected
        assert_eq!(
            output.predicted_label, *expected,
            "Case {i}: Expected {expected}, got {}",
            output.predicted_label
        );
    }

    println!("\n✅ All test cases passed!");
}
