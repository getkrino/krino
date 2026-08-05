//! Groundedness checking with ONNX DeBERTa backend.
//!
//! This example demonstrates how to use the groundedness module with the
//! production ONNX NLI model for RAG (Retrieval-Augmented Generation) verification.
//!
//! # What is Groundedness?
//!
//! Groundedness (also called faithfulness) verifies that every claim in an LLM's
//! output is supported by the provided context/source documents. This is critical
//! for RAG systems where the LLM must not hallucinate facts beyond what's in the context.
//!
//! # How it works
//!
//! 1. Split the output into individual claims (sentences)
//! 2. Split the context into sentences
//! 3. For each claim, check if ANY context sentence entails it using NLI
//! 4. Aggregate results into a faithfulness score
//!
//! # Running this example
//!
//! ```bash
//! # 1. Download the embedding model (required for performance)
//! bash scripts/download_embedding_model.sh
//!
//! # 2. Make sure you have the ONNX NLI model exported
//! cd scripts && uv run export_deberta_onnx.py
//!
//! # 3. Run the example
//! cargo run --example groundedness_onnx --release
//! ```
//!
//! # Performance Note
//!
//! This example will use CandleEmbeddingBackend if the model is downloaded,
//! otherwise it falls back to MockEmbedding (which disables pre-filtering
//! and results in 180× more NLI calls).

use krino::error::{ModelError, Result};
use krino::models::backends::candle::CandleEmbeddingBackend;
use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::{EmbeddingSimilarity, SequenceClassifier};
use krino::modules::groundedness::{GroundednessChecker, GroundednessConfig};
use std::path::Path;
use std::sync::Arc;

/// Fallback mock embedding backend (for demonstration only).
/// ⚠️ WARNING: This produces garbage similarity scores.
/// Pre-filtering will be disabled (top_k_context must be 0) which causes
/// 180× more NLI calls. Always use CandleEmbeddingBackend in production!
struct MockEmbedding;

impl EmbeddingSimilarity for MockEmbedding {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Character frequency embedding (simple similarity metric)
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
        26
    }

    fn device_info(&self) -> String {
        "MockEmbedding".to_string()
    }
}

fn main() -> Result<()> {
    // Initialize logging - silence ONNX Runtime's verbose BFC arena logs
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "info,ort::logging=warn", // Suppress ORT's internal allocator logs
        ))
        .init();

    println!("🔍 Krino Groundedness Checking Example");
    println!("=====================================\n");

    // Step 1: Load ONNX NLI model
    let model_path = Path::new("models/deberta-nli-onnx");
    if !model_path.join("model.onnx").exists() {
        eprintln!("❌ ONNX model not found at {}", model_path.display());
        eprintln!("   Run: cd scripts && uv run export_deberta_onnx.py");
        return Err(ModelError::load_failed(model_path, "Model not found").into());
    }

    println!(
        "Loading ONNX DeBERTa NLI model from {}...",
        model_path.display()
    );
    let nli_backend = Arc::new(OnnxSequenceClassifier::from_pretrained(model_path)?);
    println!("✓ Model loaded");
    println!("  Device: {}", nli_backend.device_info());
    println!("  Labels: {:?}\n", nli_backend.label_map());

    // Step 2: Load embedding model (for pre-filtering)
    println!("Loading sentence-transformers embedding model...");
    let embedding_path = Path::new("models/all-MiniLM-L6-v2");

    let (embedding_backend, use_prefilter): (Arc<dyn EmbeddingSimilarity>, bool) =
        if embedding_path.join("model.safetensors").exists() {
            println!("✓ Using CandleEmbeddingBackend (all-MiniLM-L6-v2)");
            println!("  This enables top-K pre-filtering for ~180× speedup\n");
            (
                Arc::new(CandleEmbeddingBackend::from_pretrained(embedding_path)?),
                true,
            )
        } else {
            println!("⚠️  Embedding model not found - falling back to MockEmbedding");
            println!("   Download it with: bash scripts/download_embedding_model.sh");
            println!("   Pre-filtering will be DISABLED (expect 18,000+ NLI calls)\n");
            (Arc::new(MockEmbedding), false)
        };

    // Step 3: Configure groundedness checker
    let config = GroundednessConfig {
        contradiction_threshold: 0.7,
        treat_neutral_as_unsupported: false,
        min_claim_length: 10,
        // Enable pre-filtering only if we have a real embedding backend
        top_k_context: if use_prefilter { 5 } else { 0 },
        min_similarity_threshold: 0.1,
        adaptive_top_k: false,
        include_entailment_matrix: true, // Include full matrix for analysis
        flag_compound_claims: true,
        partial_threshold: 0.2,
        partial_neutral_ceiling: 0.65,
        partial_similarity_floor: 0.7,
    };

    let checker = GroundednessChecker::new(nli_backend, embedding_backend, config);

    // Example 1: Fully grounded RAG output
    println!("📋 Example 1: Fully Grounded Output");
    println!("-----------------------------------");

    let context1 = "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. \
                     It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair. \
                     The tower is 330 meters tall and was the tallest man-made structure in the world until 1930.";

    let output1 = "The Eiffel Tower was built between 1887 and 1889 for the World's Fair. \
                   It is located in Paris and stands 330 meters tall.";

    println!("Context: {context1}\n");
    println!("Output: {output1}\n");

    let result1 = checker.check(context1, output1)?;

    println!("Results:");
    println!(
        "  Faithfulness score: {:.1}%",
        result1.faithfulness_score * 100.0
    );
    println!("  Total claims: {}", result1.total_claims);
    println!(
        "  Supported: {}, Neutral: {}, Contradicted: {}",
        result1.supported_claims, result1.neutral_claims, result1.contradicted_claims
    );

    for (i, verdict) in result1.verdicts.iter().enumerate() {
        println!("\n  Claim {}: \"{}\"", i + 1, verdict.claim);
        println!(
            "    Label: {} (confidence: {:.3})",
            verdict.label, verdict.entailment_prob
        );
        if let Some(evidence) = &verdict.best_evidence {
            println!(
                "    Best evidence: \"{}...\"",
                &evidence.sentence[..evidence.sentence.len().min(60)]
            );
        }
    }

    // Example 2: Hallucinated output
    println!("\n\n📋 Example 2: Hallucinated Output");
    println!("----------------------------------");

    let context2 = "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. \
                    Elon Musk joined as chairman of the board in 2004.";

    let output2 = "Tesla was founded by Elon Musk in 2003. \
                   He revolutionized the electric vehicle industry.";

    println!("Context: {context2}\n");
    println!("Output: {output2}\n");

    let result2 = checker.check(context2, output2)?;

    println!("Results:");
    println!(
        "  Faithfulness score: {:.1}%",
        result2.faithfulness_score * 100.0
    );
    println!("  Total claims: {}", result2.total_claims);
    println!(
        "  Supported: {}, Neutral: {}, Contradicted: {}",
        result2.supported_claims, result2.neutral_claims, result2.contradicted_claims
    );

    for (i, verdict) in result2.verdicts.iter().enumerate() {
        println!("\n  Claim {}: \"{}\"", i + 1, verdict.claim);
        println!(
            "    Label: {} (confidence: {:.3})",
            verdict.label, verdict.entailment_prob
        );
        println!(
            "    Supported: {}",
            if verdict.supported { "✓" } else { "✗" }
        );

        if let Some(contradiction) = &verdict.strongest_contradiction {
            println!("    ⚠️  Contradicted by: \"{}\"", contradiction.sentence);
            println!(
                "        (contradiction prob: {:.3})",
                contradiction.contradiction_prob
            );
        }
    }

    // Performance summary
    println!("\n\n⚡ Performance Summary");
    println!("---------------------");
    println!("Example 1:");
    println!("  Latency: {:.1}ms", result1.latency_ms);
    println!("  NLI calls: {}", result1.nli_calls);

    println!("\nExample 2:");
    println!("  Latency: {:.1}ms", result2.latency_ms);
    println!("  NLI calls: {}", result2.nli_calls);

    println!("\n✅ Example complete!");

    Ok(())
}
