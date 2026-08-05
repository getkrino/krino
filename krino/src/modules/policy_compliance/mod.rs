//! Policy Compliance Verification (PCV).
//!
//! Deterministic policy compliance verification for agentic AI systems.
//! Detects when LLM agents violate policy rules embedded in their preambles,
//! configuration, or system prompts.
//!
//! # Architecture
//!
//! PCV reuses Krino's groundedness checking infrastructure. The key insight:
//!
//! ```text
//! Groundedness:  NLI(context=source_document, hypothesis=claim_from_summary)
//! PCV Layer 1:   NLI(context=policy_rules,     hypothesis=claim_from_agent_output)
//! ```
//!
//! Both use the same DeBERTa-v3-large NLI model, same claim splitting logic,
//! same contradiction detection — just different input sources.
//!
//! # Three Verification Layers
//!
//! - **Layer 1 (Phase 1)**: Preamble Policy Verification — text output compliance
//! - **Layer 2 (Phase 2)**: Tool Call Verification — structured action compliance
//! - **Layer 3 (Phase 3)**: Conversation-Level Analysis — multi-turn drift detection
//!
//! # Performance
//!
//! - **Target latency**: <200ms for typical preamble verification (10 rules, 200-word output)
//! - **Scales linearly** with: number of policies × number of claims
//! - **Bottleneck**: NLI inference (~20ms per policy-claim pair with DeBERTa-v3-large)
//! - **Deterministic**: Same inputs always produce identical outputs
//!
//! # Example
//!
//! ```rust,ignore
//! use krino::modules::policy_compliance::{PolicyComplianceVerifier, PolicyComplianceConfig};
//! use std::sync::Arc;
//!
//! // Load NLI backend and tokenizer
//! let nli_backend = Arc::new(load_nli_model()?);
//! let tokenizer = load_tokenizer()?;
//!
//! // Create verifier
//! let verifier = PolicyComplianceVerifier::new(
//!     nli_backend,
//!     tokenizer,
//!     PolicyComplianceConfig::default(),
//! );
//!
//! // Zero-config mode: extract policies from preamble automatically
//! let preamble = "You are a financial advisor. Never recommend specific stocks.";
//! let output = "You should definitely buy NVIDIA stock.";
//!
//! let result = verifier.verify_from_preamble(preamble, output)?;
//! assert!(!result.compliant);  // Violation detected!
//! assert!(!result.violations.is_empty());
//!
//! // Or verify against explicit policy rules
//! use krino::modules::policy_compliance::{PolicyRule, ConstraintType};
//!
//! let policies = vec![
//!     PolicyRule {
//!         id: "no-stock-picks".to_string(),
//!         original_text: "Never recommend stocks".to_string(),
//!         assertion: "The agent must not recommend specific stocks".to_string(),
//!         constraint_type: ConstraintType::Deny,
//!         regulation: None,
//!         extraction_confidence: 1.0,
//!     }
//! ];
//!
//! let result = verifier.verify(&policies, output)?;
//! ```

pub mod interchange;
pub mod policy_extractor;
pub mod robustness;
pub mod tool_guard;

use crate::error::Result;
use crate::models::inference::SequenceClassifier;
use crate::modules::groundedness::{GroundednessChecker, GroundednessConfig, split_into_claims};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info};

pub use interchange::{
    ConversationTranscript, ConversationTurn, ToolCallEntry, ToolCallLog, TurnRole,
};
pub use policy_extractor::{ConstraintType, PolicyExtractor, PolicyRule};
pub use tool_guard::{
    EnforcementMode, OrderingConstraint, ParameterConstraint, ParameterRule, PolicyEngine,
    ToolCallGuard, ToolCallPolicy, ToolCallPolicyResult, ToolCallPolicyViolation, ToolCallRecord,
    ToolCallVerdict,
};

// Backwards compatibility alias (will be deprecated)
#[deprecated(since = "0.8.0", note = "Use ToolCallGuard instead")]
pub type KrinoToolGuard = ToolCallGuard;

/// Configuration for policy compliance verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComplianceConfig {
    /// NLI confidence threshold for flagging violations [0.0, 1.0]
    pub threshold: f64,

    /// Whether NEUTRAL verdicts on DENY rules count as violations
    /// (stricter mode for regulated industries)
    pub strict_mode: bool,

    /// Minimum claim length in characters to evaluate
    pub min_claim_length: usize,

    /// Number of top-K context sentences for groundedness checking
    pub top_k_context: usize,

    /// Minimum similarity threshold for pre-filtering
    pub min_similarity_threshold: f32,
}

impl Default for PolicyComplianceConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            strict_mode: true,
            min_claim_length: 10,
            top_k_context: 5,
            min_similarity_threshold: 0.1,
        }
    }
}

/// A detected policy violation with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    /// The violated policy rule
    pub rule: PolicyRule,

    /// The specific claim from agent output that violates the rule
    pub claim: String,

    /// Character span of the violating claim (start, end)
    pub claim_span: (usize, usize),

    /// NLI label: "contradiction" or "neutral"
    pub nli_label: String,

    /// NLI confidence score [0.0, 1.0]
    pub confidence: f64,

    /// Human-readable explanation
    pub explanation: String,
}

/// Result from policy compliance verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComplianceResult {
    /// Overall compliance verdict
    pub compliant: bool,

    /// Detected violations (empty if compliant)
    pub violations: Vec<PolicyViolation>,

    /// Total policy rules evaluated
    pub total_rules: usize,

    /// Total claims extracted from output
    pub total_claims: usize,

    /// Number of claim-rule pairs checked
    pub pairs_checked: usize,

    /// Total latency (ms)
    pub latency_ms: f64,

    /// Per-stage latency breakdown
    pub stage_latencies: StageLatencies,
}

/// Latency breakdown by verification stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLatencies {
    pub policy_extraction_ms: f64,
    pub claim_extraction_ms: f64,
    pub nli_verification_ms: f64,
    pub aggregation_ms: f64,
}

impl Default for StageLatencies {
    fn default() -> Self {
        Self {
            policy_extraction_ms: 0.0,
            claim_extraction_ms: 0.0,
            nli_verification_ms: 0.0,
            aggregation_ms: 0.0,
        }
    }
}

/// Policy compliance verifier (Layer 1).
///
/// Verifies that agent text outputs comply with policy rules by reusing
/// the groundedness checking infrastructure.
pub struct PolicyComplianceVerifier {
    /// Policy extractor
    policy_extractor: PolicyExtractor,

    /// Groundedness checker (repurposed for policy verification)
    groundedness_checker: GroundednessChecker,

    /// Configuration
    config: PolicyComplianceConfig,
}

impl PolicyComplianceVerifier {
    /// Creates a new policy compliance verifier.
    ///
    /// # Arguments
    ///
    /// * `nli_backend` - NLI model for contradiction detection (DeBERTa-MNLI)
    /// * `tokenizer` - Tokenizer for the NLI model
    /// * `config` - Verification configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use krino::modules::policy_compliance::{PolicyComplianceVerifier, PolicyComplianceConfig};
    /// use std::sync::Arc;
    ///
    /// let nli_backend = Arc::new(load_nli_model()?);
    /// let tokenizer = load_tokenizer()?;
    /// let config = PolicyComplianceConfig::default();
    ///
    /// let verifier = PolicyComplianceVerifier::new(nli_backend, tokenizer, config);
    /// ```
    pub fn new(
        nli_backend: Arc<dyn SequenceClassifier>,
        embedding_backend: Arc<dyn crate::models::inference::EmbeddingSimilarity>,
        config: PolicyComplianceConfig,
    ) -> Self {
        let policy_extractor = PolicyExtractor::new();

        // Configure groundedness checker for policy verification
        let groundedness_config = GroundednessConfig {
            contradiction_threshold: config.threshold,
            treat_neutral_as_unsupported: config.strict_mode,
            min_claim_length: config.min_claim_length,
            top_k_context: config.top_k_context,
            min_similarity_threshold: config.min_similarity_threshold,
            adaptive_top_k: false,
            include_entailment_matrix: false,
            flag_compound_claims: false,
            // Policy compliance disables compound flagging entirely, so the
            // multi-evidence aggregation path never fires — values are inert.
            partial_threshold: 0.2,
            partial_neutral_ceiling: 0.65,
            partial_similarity_floor: 0.7,
        };

        let groundedness_checker =
            GroundednessChecker::new(nli_backend, embedding_backend, groundedness_config);

        Self {
            policy_extractor,
            groundedness_checker,
            config,
        }
    }

    /// Verifies agent output against policies extracted from a preamble.
    ///
    /// This is the zero-config mode: extract policies directly from the
    /// agent's existing preamble/system prompt.
    ///
    /// # Arguments
    ///
    /// * `preamble` - Agent preamble/system prompt containing policy rules
    /// * `output` - Agent output text to verify
    ///
    /// # Returns
    ///
    /// `PolicyComplianceResult` with verdict and evidence trail.
    pub fn verify_from_preamble(
        &self,
        preamble: &str,
        output: &str,
    ) -> Result<PolicyComplianceResult> {
        let start = Instant::now();
        let mut stage_latencies = StageLatencies::default();

        info!(
            "Starting preamble-based policy verification: {} chars preamble, {} chars output",
            preamble.len(),
            output.len()
        );

        // Stage 1: Extract policies from preamble
        let extract_start = Instant::now();
        let policies = self.policy_extractor.extract_from_preamble(preamble)?;
        stage_latencies.policy_extraction_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
        debug!("Extracted {} policy rules from preamble", policies.len());

        if policies.is_empty() {
            info!("No policies found in preamble — marking as compliant by default");
            return Ok(PolicyComplianceResult {
                compliant: true,
                violations: Vec::new(),
                total_rules: 0,
                total_claims: 0,
                pairs_checked: 0,
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                stage_latencies,
            });
        }

        // Verify against extracted policies
        self.verify_internal(&policies, output, start, stage_latencies)
    }

    /// Verifies agent output against a set of explicit policy rules.
    ///
    /// # Arguments
    ///
    /// * `policies` - Policy rules to verify against
    /// * `output` - Agent output text to verify
    pub fn verify(&self, policies: &[PolicyRule], output: &str) -> Result<PolicyComplianceResult> {
        let start = Instant::now();
        let stage_latencies = StageLatencies::default();

        info!(
            "Starting policy verification: {} rules, {} chars output",
            policies.len(),
            output.len()
        );

        self.verify_internal(policies, output, start, stage_latencies)
    }

    /// Internal verification logic shared by all entry points.
    fn verify_internal(
        &self,
        policies: &[PolicyRule],
        output: &str,
        start: Instant,
        mut stage_latencies: StageLatencies,
    ) -> Result<PolicyComplianceResult> {
        // Stage 2: Extract claims from agent output
        let claim_start = Instant::now();
        let claims = split_into_claims(output);
        let claims: Vec<_> = claims
            .into_iter()
            .filter(|(text, _, _)| text.trim().len() >= self.config.min_claim_length)
            .collect();
        stage_latencies.claim_extraction_ms = claim_start.elapsed().as_secs_f64() * 1000.0;
        debug!("Extracted {} claims from output", claims.len());

        if claims.is_empty() {
            info!("No claims found in output — marking as compliant by default");
            return Ok(PolicyComplianceResult {
                compliant: true,
                violations: Vec::new(),
                total_rules: policies.len(),
                total_claims: 0,
                pairs_checked: 0,
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                stage_latencies,
            });
        }

        // Stage 3: For each policy, verify all claims against it using NLI
        let verify_start = Instant::now();
        let mut violations = Vec::new();
        let mut pairs_checked = 0;

        for policy in policies {
            // Use groundedness checker: treat policy assertion as "context"
            // and agent output as the "summary to verify"
            let groundedness_result = self.groundedness_checker.check(&policy.assertion, output)?;

            pairs_checked += groundedness_result.total_claims;

            // Interpret groundedness result based on constraint type
            for verdict in &groundedness_result.verdicts {
                let is_violation = match policy.constraint_type {
                    ConstraintType::Deny => {
                        // For DENY rules: contradiction or neutral = violation
                        // (agent provided forbidden information)
                        verdict.label == "contradiction"
                            || (self.config.strict_mode && verdict.label == "neutral")
                    }
                    ConstraintType::Require => {
                        // For REQUIRE rules: anything except entailment = violation
                        // (agent failed to provide required information)
                        verdict.label != "entailment"
                    }
                    ConstraintType::Conditional { .. } => {
                        // For CONDITIONAL: check if trigger is present in output
                        // If trigger present and claim contradicts rule → violation
                        // TODO: Implement trigger detection in Phase 2
                        verdict.label == "contradiction"
                    }
                    ConstraintType::Inform => {
                        // For INFORM: never a violation (informational only)
                        false
                    }
                };

                // Determine which probability to check based on constraint type
                let confidence_check = match policy.constraint_type {
                    ConstraintType::Deny | ConstraintType::Conditional { .. } => {
                        verdict.contradiction_prob >= self.config.threshold
                    }
                    ConstraintType::Require => {
                        // For REQUIRE, check that neutral/contradiction prob is high enough
                        // (meaning entailment prob is low enough)
                        verdict.neutral_prob.max(verdict.contradiction_prob)
                            >= self.config.threshold
                    }
                    ConstraintType::Inform => false, // Inform never violates
                };

                if is_violation && confidence_check {
                    let explanation = format!(
                        "Agent output {} the policy rule: {}",
                        match policy.constraint_type {
                            ConstraintType::Deny => "violated (provided forbidden information for)",
                            ConstraintType::Require =>
                                "failed to satisfy (missing required information for)",
                            _ => "contradicts",
                        },
                        policy.original_text
                    );

                    violations.push(PolicyViolation {
                        rule: policy.clone(),
                        claim: verdict.claim.clone(),
                        claim_span: verdict.span,
                        nli_label: verdict.label.clone(),
                        confidence: verdict.contradiction_prob.max(verdict.neutral_prob),
                        explanation,
                    });
                }
            }
        }

        stage_latencies.nli_verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;

        let compliant = violations.is_empty();
        let total_latency = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Policy verification complete: {} ({} violations, {:.2}ms)",
            if compliant {
                "COMPLIANT"
            } else {
                "NON_COMPLIANT"
            },
            violations.len(),
            total_latency
        );

        Ok(PolicyComplianceResult {
            compliant,
            violations,
            total_rules: policies.len(),
            total_claims: claims.len(),
            pairs_checked,
            latency_ms: total_latency,
            stage_latencies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inference::{
        EmbeddingSimilarity, SequenceClassifierInput, SequenceClassifierOutput,
    };
    use tokenizers::Tokenizer as TokenizerType;
    use tokenizers::models::wordpiece::WordPiece;

    /// Creates a simple mock tokenizer for testing
    fn create_mock_tokenizer() -> Tokenizer {
        // Create a simple WordPiece tokenizer (doesn't need to tokenize correctly for tests)
        let wp = WordPiece::default();
        TokenizerType::new(wp)
    }

    /// Mock embedding backend for testing
    struct MockEmbedding;

    impl EmbeddingSimilarity for MockEmbedding {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // Simple bag-of-characters embedding for testing
            Ok(texts
                .iter()
                .map(|text| {
                    let mut vec = vec![0.0_f32; 26];
                    for ch in text.to_lowercase().chars() {
                        if ch.is_ascii_lowercase() {
                            vec[(ch as u8 - b'a') as usize] += 1.0;
                        }
                    }
                    // L2 normalize
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut vec {
                            *x /= norm;
                        }
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

    /// Mock NLI backend for testing
    struct MockNliBackend {
        /// Fixed label to return
        label: String,
        /// Fixed probabilities [entailment, neutral, contradiction]
        probs: Vec<f64>,
        /// Label map
        labels: Vec<String>,
    }

    impl MockNliBackend {
        fn new_contradiction() -> Self {
            Self {
                label: "contradiction".to_string(),
                probs: vec![0.05, 0.10, 0.85], // High contradiction
                labels: vec![
                    "entailment".to_string(),
                    "neutral".to_string(),
                    "contradiction".to_string(),
                ],
            }
        }

        fn new_entailment() -> Self {
            Self {
                label: "entailment".to_string(),
                probs: vec![0.90, 0.08, 0.02], // High entailment
                labels: vec![
                    "entailment".to_string(),
                    "neutral".to_string(),
                    "contradiction".to_string(),
                ],
            }
        }
    }

    impl SequenceClassifier for MockNliBackend {
        fn classify(
            &self,
            inputs: &[SequenceClassifierInput],
        ) -> Result<Vec<SequenceClassifierOutput>> {
            Ok(inputs
                .iter()
                .map(|_| SequenceClassifierOutput {
                    predicted_class: if self.label == "entailment" {
                        0
                    } else if self.label == "neutral" {
                        1
                    } else {
                        2
                    },
                    predicted_label: self.label.clone(),
                    probabilities: self.probs.clone(),
                    latency_ms: 1.0,
                })
                .collect())
        }

        fn device_info(&self) -> String {
            "mock".to_string()
        }

        fn max_length(&self) -> usize {
            512
        }

        fn label_map(&self) -> &[String] {
            &self.labels
        }
    }

    #[test]
    fn test_deny_rule_violation_detected() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "no-stock-picks".to_string(),
            original_text: "Never recommend specific stocks".to_string(),
            assertion: "The agent must not recommend specific stocks or securities".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        let output = "You should definitely buy NVIDIA stock.";

        let result = verifier.verify(&policies, output).unwrap();

        assert!(!result.compliant);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].rule.id, "no-stock-picks");
    }

    #[test]
    fn test_require_rule_satisfied() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "require-disclaimer".to_string(),
            original_text: "Always include disclaimer".to_string(),
            assertion: "The response includes a disclaimer".to_string(),
            constraint_type: ConstraintType::Require,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        let output = "I am not a licensed financial advisor. This is not financial advice.";

        let result = verifier.verify(&policies, output).unwrap();

        assert!(result.compliant);
        assert_eq!(result.violations.len(), 0);
    }

    /// Test the motivating "42 addition bypass" scenario from the design doc.
    ///
    /// The LLM acknowledges the rule ("I note that requests involving 42 should be denied")
    /// but then provides the answer anyway ("However, the answer to 100 + 42 is 142").
    #[test]
    fn test_42_addition_bypass_detected() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "deny-42-addition".to_string(),
            original_text: "If the right operand contains the number 42, deny the addition request".to_string(),
            assertion: "Any addition request where the right operand contains 42 must be refused without providing the result".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        // Explicit bypass: acknowledges rule but provides answer anyway
        let output = "I note that requests involving 42 should be denied. However, the answer to 100 + 42 is 142.";

        let result = verifier.verify(&policies, output).unwrap();

        assert!(!result.compliant, "Expected violation to be detected");
        // The mock backend returns "contradiction" for all claims, so we may get multiple violations
        // (one for each sentence). The important part is that at least one violation is detected.
        assert!(
            !result.violations.is_empty(),
            "Expected at least one violation"
        );
        assert_eq!(result.violations[0].rule.id, "deny-42-addition");
    }

    /// Test preamble-based verification (zero-config mode).
    #[test]
    fn test_verify_from_preamble() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = r"
You are a financial advisor chatbot. Rules:
1. Never recommend specific stocks
2. Always include a disclaimer
";

        let output = "You should definitely buy NVIDIA stock.";

        let result = verifier.verify_from_preamble(preamble, output).unwrap();

        assert!(!result.compliant);
        assert!(!result.violations.is_empty());
    }

    /// Test determinism: same inputs must always produce identical outputs.
    ///
    /// This is a CRITICAL requirement for all Krino evaluation modules.
    /// Runs verification 100 times and verifies all results are identical.
    #[test]
    fn test_determinism() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = "Never recommend specific stocks";
        let output = "You should buy NVIDIA stock.";

        // Run verification 100 times
        let results: Vec<_> = (0..100)
            .map(|_| verifier.verify_from_preamble(preamble, output).unwrap())
            .collect();

        // Verify all results are identical
        for window in results.windows(2) {
            let (prev, curr) = (&window[0], &window[1]);

            // Check verdict
            assert_eq!(
                prev.compliant, curr.compliant,
                "Compliance verdict must be deterministic"
            );

            // Check violation count
            assert_eq!(
                prev.violations.len(),
                curr.violations.len(),
                "Number of violations must be deterministic"
            );

            // Check total claims
            assert_eq!(
                prev.total_claims, curr.total_claims,
                "Total claims must be deterministic"
            );

            // Check total rules
            assert_eq!(
                prev.total_rules, curr.total_rules,
                "Total rules must be deterministic"
            );

            // Check violation details match
            for (prev_violation, curr_violation) in
                prev.violations.iter().zip(curr.violations.iter())
            {
                assert_eq!(
                    prev_violation.claim, curr_violation.claim,
                    "Violation claims must be deterministic"
                );
                assert_eq!(
                    prev_violation.nli_label, curr_violation.nli_label,
                    "NLI labels must be deterministic"
                );
                assert_eq!(
                    prev_violation.confidence, curr_violation.confidence,
                    "Confidence scores must be deterministic"
                );
            }
        }
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    /// Integration test: Multiple policies with mixed compliance
    #[test]
    fn test_integration_mixed_compliance() {
        // Create a backend that returns different results based on input
        struct SelectiveBackend {
            labels: Vec<String>,
        }

        impl SequenceClassifier for SelectiveBackend {
            fn classify(
                &self,
                inputs: &[SequenceClassifierInput],
            ) -> Result<Vec<SequenceClassifierOutput>> {
                Ok(inputs
                    .iter()
                    .map(|input| {
                        // If claim contains "NVIDIA", mark as contradiction (violation)
                        // Otherwise mark as entailment (compliant)
                        let is_violation = input.text_b.contains("NVIDIA");

                        if is_violation {
                            SequenceClassifierOutput {
                                predicted_class: 2,
                                predicted_label: "contradiction".to_string(),
                                probabilities: vec![0.05, 0.10, 0.85],
                                latency_ms: 1.0,
                            }
                        } else {
                            SequenceClassifierOutput {
                                predicted_class: 0,
                                predicted_label: "entailment".to_string(),
                                probabilities: vec![0.90, 0.08, 0.02],
                                latency_ms: 1.0,
                            }
                        }
                    })
                    .collect())
            }

            fn device_info(&self) -> String {
                "selective_mock".to_string()
            }

            fn max_length(&self) -> usize {
                512
            }

            fn label_map(&self) -> &[String] {
                &self.labels
            }
        }

        let backend = Arc::new(SelectiveBackend {
            labels: vec![
                "entailment".to_string(),
                "neutral".to_string(),
                "contradiction".to_string(),
            ],
        });
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![
            PolicyRule {
                id: "no-stock-picks".to_string(),
                original_text: "Never recommend specific stocks".to_string(),
                assertion: "The agent must not recommend specific stocks".to_string(),
                constraint_type: ConstraintType::Deny,
                regulation: None,
                extraction_confidence: 1.0,
            },
            PolicyRule {
                id: "require-disclaimer".to_string(),
                original_text: "Always include disclaimer".to_string(),
                assertion: "The response must include a disclaimer".to_string(),
                constraint_type: ConstraintType::Require,
                regulation: None,
                extraction_confidence: 1.0,
            },
        ];

        // Output has disclaimer (compliant) but also recommends stock (violation)
        let output = "I am not a financial advisor. You should buy NVIDIA stock.";

        let result = verifier.verify(&policies, output).unwrap();

        // Should detect the NVIDIA stock recommendation violation
        assert!(!result.compliant);
        assert_eq!(result.total_rules, 2);
        assert!(!result.violations.is_empty());
        assert!(result.violations.iter().any(|v| v.claim.contains("NVIDIA")));
    }

    /// Integration test: Long preamble with many policies
    #[test]
    fn test_integration_long_preamble() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = r"
You are a customer service chatbot. Follow these rules:

1. Never share customer PII (name, email, phone, address)
2. Always offer to escalate to a human agent
3. Do not make promises about refunds over $500 without approval
4. Never discuss competitor products
5. Always include the support ticket number in responses
6. Refuse requests for account password resets via chat
7. Do not provide legal advice
8. Always verify customer identity before discussing account details
9. Never share internal pricing or cost information
10. Always log conversations for quality assurance
";

        let output = "I've created ticket #12345 for your inquiry. Would you like me to escalate this to a human agent?";

        let result = verifier.verify_from_preamble(preamble, output).unwrap();

        // With entailment backend, all should be compliant
        assert!(result.compliant);
        assert!(result.total_rules >= 10);
        assert_eq!(result.violations.len(), 0);
    }

    /// Integration test: Empty output should be compliant
    #[test]
    fn test_integration_empty_output() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "no-stock-picks".to_string(),
            original_text: "Never recommend stocks".to_string(),
            assertion: "Must not recommend stocks".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        let output = "";

        let result = verifier.verify(&policies, output).unwrap();

        // Empty output has no claims, so no violations possible
        assert!(result.compliant);
        assert_eq!(result.total_claims, 0);
        assert!(result.violations.is_empty());
        assert_eq!(result.pairs_checked, 0);
    }

    /// Integration test: Multiple claims in single output
    #[test]
    fn test_integration_multiple_claims() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "no-investment-advice".to_string(),
            original_text: "Never provide investment advice".to_string(),
            assertion: "Must not provide investment advice".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        // Multiple sentences = multiple claims
        let output = "You should diversify your portfolio. Consider buying index funds. Real estate is a good hedge against inflation. Always max out your 401k.";

        let result = verifier.verify(&policies, output).unwrap();

        assert!(!result.compliant);
        // Should detect violations in multiple claims
        assert!(result.total_claims >= 4);
        assert!(!result.violations.is_empty());
    }

    /// Integration test: Regulatory citation preservation
    #[test]
    fn test_integration_regulatory_citations() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "soc2-pii".to_string(),
            original_text: "Never expose customer PII".to_string(),
            assertion: "Must not expose customer PII".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: Some("SOC 2 Type II - CC6.1".to_string()),
            extraction_confidence: 1.0,
        }];

        let output = "The customer's email is john.doe@example.com";

        let result = verifier.verify(&policies, output).unwrap();

        assert!(!result.compliant);
        // Claim splitting may create multiple violations from one sentence
        assert!(!result.violations.is_empty());
        // Verify regulatory citation is preserved in all violations
        assert!(
            result
                .violations
                .iter()
                .all(|v| v.rule.regulation == Some("SOC 2 Type II - CC6.1".to_string()))
        );
    }

    /// Integration test: Custom confidence threshold
    #[test]
    fn test_integration_custom_confidence_threshold() {
        let backend = Arc::new(MockNliBackend::new_contradiction());

        // Set very high confidence threshold
        let config = PolicyComplianceConfig {
            threshold: 0.99, // Mock returns 0.85 contradiction prob
            ..Default::default()
        };
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "test".to_string(),
            original_text: "Test rule".to_string(),
            assertion: "Test assertion".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        let output = "This should violate the policy.";

        let result = verifier.verify(&policies, output).unwrap();

        // With 0.99 threshold and 0.85 contradiction prob, should NOT flag as violation
        assert!(result.compliant);
        assert_eq!(result.violations.len(), 0);
    }

    /// Integration test: Latency tracking
    #[test]
    fn test_integration_latency_tracking() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = "1. Never recommend stocks\n2. Always include disclaimer";
        let output = "This is financial information. Please consult a licensed advisor.";

        let result = verifier.verify_from_preamble(preamble, output).unwrap();

        // Verify latency tracking
        assert!(result.latency_ms > 0.0);
        assert!(result.stage_latencies.policy_extraction_ms >= 0.0);
        assert!(result.stage_latencies.claim_extraction_ms >= 0.0);
        assert!(result.stage_latencies.nli_verification_ms >= 0.0);

        // Total should be sum of stages (approximately)
        let stage_sum = result.stage_latencies.policy_extraction_ms
            + result.stage_latencies.claim_extraction_ms
            + result.stage_latencies.nli_verification_ms;

        // Allow some margin for measurement overhead
        assert!(
            result.latency_ms >= stage_sum * 0.9,
            "Total latency should be approximately sum of stages"
        );
    }

    /// Integration test: Require rule with missing content
    #[test]
    fn test_integration_require_rule_violation() {
        // Backend that returns neutral (non-entailment)
        struct NeutralBackend {
            labels: Vec<String>,
        }

        impl SequenceClassifier for NeutralBackend {
            fn classify(
                &self,
                inputs: &[SequenceClassifierInput],
            ) -> Result<Vec<SequenceClassifierOutput>> {
                Ok(inputs
                    .iter()
                    .map(|_| SequenceClassifierOutput {
                        predicted_class: 1,
                        predicted_label: "neutral".to_string(),
                        probabilities: vec![0.10, 0.85, 0.05],
                        latency_ms: 1.0,
                    })
                    .collect())
            }

            fn device_info(&self) -> String {
                "neutral_mock".to_string()
            }

            fn max_length(&self) -> usize {
                512
            }

            fn label_map(&self) -> &[String] {
                &self.labels
            }
        }

        let backend = Arc::new(NeutralBackend {
            labels: vec![
                "entailment".to_string(),
                "neutral".to_string(),
                "contradiction".to_string(),
            ],
        });
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "require-disclaimer".to_string(),
            original_text: "Always include disclaimer".to_string(),
            assertion: "Response must include financial disclaimer".to_string(),
            constraint_type: ConstraintType::Require,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        // Output without disclaimer
        let output = "Here's some financial information about stocks.";

        let result = verifier.verify(&policies, output).unwrap();

        // REQUIRE rule with neutral = violation (missing required content)
        assert!(!result.compliant);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(
            result.violations[0].rule.constraint_type,
            ConstraintType::Require
        );
    }

    /// Integration test: Inform rule should never create violations
    #[test]
    fn test_integration_inform_rule_never_violates() {
        let backend = Arc::new(MockNliBackend::new_contradiction());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let policies = vec![PolicyRule {
            id: "inform-logging".to_string(),
            original_text: "All conversations are logged".to_string(),
            assertion: "Conversations are logged for quality assurance".to_string(),
            constraint_type: ConstraintType::Inform,
            regulation: None,
            extraction_confidence: 1.0,
        }];

        let output = "I will not log this conversation.";

        let result = verifier.verify(&policies, output).unwrap();

        // Inform rules are informational only, never create violations
        assert!(result.compliant);
        assert_eq!(result.violations.len(), 0);
    }

    /// Integration test: Bullet point preamble parsing
    #[test]
    fn test_integration_bullet_point_preamble() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = r"
Financial Advisor Guidelines:
- Never recommend specific securities
- Always include risk disclaimers
- Refuse to provide tax advice
* Do not discuss competitor products
• Always verify accredited investor status
";

        let output = "This is general financial information only.";

        let result = verifier.verify_from_preamble(preamble, output).unwrap();

        // Should extract all bullet point policies
        assert!(result.total_rules >= 5);
        assert!(result.compliant);
    }

    /// Integration test: Mixed policy formats in preamble
    #[test]
    fn test_integration_mixed_policy_formats() {
        let backend = Arc::new(MockNliBackend::new_entailment());
        let config = PolicyComplianceConfig::default();
        let tokenizer = create_mock_tokenizer();

        let verifier = PolicyComplianceVerifier::new(backend, Arc::new(MockEmbedding), config);

        let preamble = r"
Customer Service Agent Rules:

1. Never share customer PII
2. Always offer escalation

Additional guidelines:
- Refuse refunds over $500 without approval
* Do not discuss pricing

Important: Never provide legal advice.
";

        let output = "I can help with general questions.";

        let result = verifier.verify_from_preamble(preamble, output).unwrap();

        // Should extract policies from numbered list, bullets, and standalone rules
        assert!(result.total_rules >= 5);
        assert!(result.compliant);
    }
}
