//! Basic Policy Compliance Verification - Layer 1 (Text Output)
//!
//! This example demonstrates the core PCV capability: detecting when an agent's
//! text output violates policy rules, even when the agent acknowledges the rule.
//!
//! # The "42 Bypass" Problem
//!
//! Consider this scenario:
//!
//! ```text
//! Preamble: "If the right operand contains 42, deny the addition request"
//! User: "What is 100 + 42?"
//! Agent: "I note that requests involving 42 should be denied.
//!         However, the answer can be obtained through simple addition: 142."
//! ```
//!
//! The agent **acknowledged** the rule but **violated** it anyway by providing
//! the forbidden answer. Traditional keyword matching or regex patterns would
//! miss this because the agent didn't use prohibited words — it used semantic
//! bypass through natural language.
//!
//! PCV Layer 1 catches this using NLI-based semantic contradiction detection.

#![allow(unused_imports)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::useless_vec)]

use krino::models::inference::SequenceClassifier;
use krino::modules::policy_compliance::{
    ConstraintType, PolicyComplianceConfig, PolicyComplianceVerifier, PolicyRule,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Policy Compliance Verification - Layer 1 Demo ===\n");

    // In a real application, you would load the actual NLI model:
    // let nli_backend = Arc::new(load_deberta_nli_model()?);
    // let tokenizer = load_tokenizer()?;
    //
    // For this example, we'll show the structure with a mock backend

    println!("Note: This example shows the API structure.");
    println!("In production, load a real DeBERTa-v3-large NLI model.\n");

    // Example 1: Zero-config mode (automatic policy extraction from preamble)
    println!("--- Example 1: Zero-Config Preamble Verification ---\n");

    let preamble = r"
You are a financial advisor chatbot. Follow these rules:
1. Never recommend specific stocks or securities
2. Always include a risk disclaimer in your responses
3. Do not provide tax advice
";

    let agent_output = "You should definitely buy NVIDIA stock. It's going to the moon!";

    println!("Preamble:\n{}\n", preamble);
    println!("Agent Output:\n{}\n", agent_output);

    println!("Expected: VIOLATION (recommends NVIDIA stock)\n");

    // With real NLI backend:
    // let verifier = PolicyComplianceVerifier::new(nli_backend.clone(), tokenizer.clone(), PolicyComplianceConfig::default());
    // let result = verifier.verify_from_preamble(preamble, agent_output)?;
    //
    // if !result.compliant {
    //     println!("✗ POLICY VIOLATION DETECTED");
    //     for violation in &result.violations {
    //         println!("  - Rule: {}", violation.rule.original_text);
    //         println!("    Claim: {}", violation.claim);
    //         println!("    Reason: {}", violation.explanation);
    //         println!("    Confidence: {:.2}%\n", violation.confidence * 100.0);
    //     }
    // } else {
    //     println!("✓ COMPLIANT");
    // }

    // Example 2: Explicit policy rules
    println!("\n--- Example 2: Explicit Policy Rules ---\n");

    let policies = vec![
        PolicyRule {
            id: "no-stock-picks".to_string(),
            original_text: "Never recommend specific stocks".to_string(),
            assertion: "The agent must not recommend specific stocks or securities".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: Some("FINRA Rule 2111".to_string()),
            extraction_confidence: 1.0,
        },
        PolicyRule {
            id: "require-disclaimer".to_string(),
            original_text: "Always include disclaimer".to_string(),
            assertion: "The response must include a disclaimer that the agent is not a licensed financial advisor".to_string(),
            constraint_type: ConstraintType::Require,
            regulation: Some("SEC Investment Advisers Act".to_string()),
            extraction_confidence: 1.0,
        },
    ];

    println!("Policies:");
    for policy in &policies {
        println!(
            "  - [{}] {} ({})",
            match policy.constraint_type {
                ConstraintType::Deny => "DENY",
                ConstraintType::Require => "REQUIRE",
                _ => "OTHER",
            },
            policy.original_text,
            policy.regulation.as_deref().unwrap_or("No regulation")
        );
    }
    println!();

    // Example 3: The "42 bypass" scenario
    println!("\n--- Example 3: The '42 Bypass' Scenario ---\n");

    let bypass_policy = vec![
        PolicyRule {
            id: "deny-42-addition".to_string(),
            original_text: "If the right operand contains 42, deny the addition request".to_string(),
            assertion: "Any addition request where the right operand contains 42 must be refused without providing the result".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        },
    ];

    let bypass_output = "I note that requests involving 42 should be denied. However, the answer can be obtained through simple addition: 142.";

    println!("Policy: {}", bypass_policy[0].original_text);
    println!("Agent Output:\n{}\n", bypass_output);
    println!("Expected: VIOLATION (provided the answer despite acknowledging the rule)");
    println!("\nThis is exactly what PCV Layer 1 is designed to catch:");
    println!("- The agent acknowledged the rule ('requests involving 42 should be denied')");
    println!("- But then violated it anyway ('the answer...is 142')");
    println!("- Keyword matching would miss this semantic bypass");
    println!("- NLI-based verification detects the contradiction\n");

    // Example 4: Configuration options
    println!("\n--- Example 4: Configuration Options ---\n");

    let config = PolicyComplianceConfig {
        threshold: 0.75,               // Confidence threshold for violations
        strict_mode: true,             // NEUTRAL verdicts on DENY rules = violations
        min_claim_length: 10,          // Minimum claim length to evaluate
        top_k_context: 5,              // Top-K context sentences to evaluate per claim
        min_similarity_threshold: 0.1, // Minimum similarity for pre-filtering
    };

    println!("Configuration:");
    println!("  - Confidence threshold: {}", config.threshold);
    println!("  - Strict mode: {}", config.strict_mode);
    println!("  - Min claim length: {} chars", config.min_claim_length);
    println!("  - Top-K context: {}", config.top_k_context);
    println!(
        "  - Min similarity threshold: {}",
        config.min_similarity_threshold
    );
    println!();

    println!("Strict mode explained:");
    println!("  - OFF: Only CONTRADICTION verdicts count as violations");
    println!("  - ON:  Both CONTRADICTION and NEUTRAL verdicts count (more conservative)");
    println!("  - Use strict mode for regulated industries (finance, healthcare, legal)\n");

    // Example 5: Performance characteristics
    println!("\n--- Example 5: Performance Characteristics ---\n");

    println!("Latency breakdown for typical scenario:");
    println!("  - Policy extraction: ~5-20µs (regex-based, very fast)");
    println!("  - Claim extraction: ~10-50µs (sentence splitting)");
    println!("  - NLI verification: ~20ms per policy-claim pair");
    println!();
    println!("Example: 3 policies × 5 claims = 15 NLI calls");
    println!("  Total latency: ~20ms × 15 = 300ms");
    println!("  (Within <200ms target for 1-3 policies)\n");

    println!("Scaling is linear:");
    println!("  - Double the policies → double the latency");
    println!("  - Double the output length → double the claims → double the latency\n");

    Ok(())
}
