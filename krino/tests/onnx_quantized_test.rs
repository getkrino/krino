//! Quick test to compare quantized vs full ONNX model performance.

use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::{SequenceClassifier, SequenceClassifierInput};
use std::path::Path;
use std::time::Instant;

#[test]
#[ignore]
fn test_quantized_model_performance() {
    let full_path = Path::new("models/deberta-nli-onnx");
    let quantized_path = Path::new("models/deberta-nli-onnx-quantized");

    // Test input
    let input = SequenceClassifierInput {
        text_a: "The cat sat on the mat.".to_string(),
        text_b: "An animal was on the mat.".to_string(),
    };

    // Test full model
    if full_path.join("model.onnx").exists() {
        println!("\n📊 Testing FULL model (1.7GB)...");
        let classifier =
            OnnxSequenceClassifier::from_pretrained(full_path).expect("Failed to load full model");

        let mut times = vec![];
        for _ in 0..10 {
            let start = Instant::now();
            let _ = classifier
                .classify(std::slice::from_ref(&input))
                .expect("Inference failed");
            times.push(start.elapsed().as_millis());
        }
        let avg = times.iter().sum::<u128>() / times.len() as u128;
        println!("  Average latency: {}ms (over 10 runs)", avg);
        println!(
            "  Min: {}ms, Max: {}ms",
            times.iter().min().unwrap(),
            times.iter().max().unwrap()
        );
    }

    // Test quantized model
    if quantized_path.join("model.onnx").exists() {
        println!("\n📊 Testing QUANTIZED model (613MB)...");
        let classifier = OnnxSequenceClassifier::from_pretrained(quantized_path)
            .expect("Failed to load quantized model");

        let mut times = vec![];
        for _ in 0..10 {
            let start = Instant::now();
            let result = classifier
                .classify(std::slice::from_ref(&input))
                .expect("Inference failed");
            times.push(start.elapsed().as_millis());

            // Verify it still works correctly
            if times.len() == 1 {
                println!(
                    "  First prediction: {} ({:.3})",
                    result[0].predicted_label, result[0].probabilities[result[0].predicted_class]
                );
            }
        }
        let avg = times.iter().sum::<u128>() / times.len() as u128;
        println!("  Average latency: {}ms (over 10 runs)", avg);
        println!(
            "  Min: {}ms, Max: {}ms",
            times.iter().min().unwrap(),
            times.iter().max().unwrap()
        );
    }
}
