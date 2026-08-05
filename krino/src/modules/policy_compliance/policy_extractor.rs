//! Policy extraction from preambles and structured formats.
//!
//! Converts human-readable policy text (from agent preambles) and structured
//! policy files (YAML/JSON) into normalized `PolicyRule` objects suitable
//! for NLI-based verification.

use crate::error::{ConfigError, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tracing::debug;

/// Type of policy constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstraintType {
    /// Agent must NOT do this (e.g., "never recommend stocks")
    Deny,
    /// Agent MUST do this (e.g., "always include disclaimer")
    Require,
    /// Agent must do this ONLY WHEN condition is met
    Conditional { trigger: String },
    /// Agent should convey this information (informational only, never fails)
    Inform,
}

/// A parsed policy rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Unique identifier
    pub id: String,

    /// Original text from preamble or policy file
    pub original_text: String,

    /// Normalized assertion for NLI
    /// e.g., "The agent must not recommend specific stocks"
    pub assertion: String,

    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Optional regulatory citation (e.g., "SEC Investment Advisers Act §202(a)(11)")
    pub regulation: Option<String>,

    /// Confidence that extraction was correct [0.0, 1.0]
    pub extraction_confidence: f64,
}

/// Policy extractor.
///
/// Parses policy rules from various formats:
/// - Agent preambles (plain text, numbered lists, bullet points)
/// - Structured YAML policy files
/// - Structured JSON policy files
pub struct PolicyExtractor {
    /// Numbered list pattern (e.g., "1. Never do X")
    numbered_pattern: Regex,
    /// Bullet point pattern (e.g., "- Always do Y")
    bullet_pattern: Regex,
}

impl PolicyExtractor {
    /// Creates a new policy extractor.
    ///
    /// # Panics
    ///
    /// Panics if the built-in regex patterns fail to compile. This indicates
    /// a bug in the library code and should never happen in practice.
    #[must_use]
    pub fn new() -> Self {
        Self {
            numbered_pattern: Regex::new(r"^\s*(\d+)\.\s+(.+)$").unwrap(),
            bullet_pattern: Regex::new(r"^\s*[-*•]\s+(.+)$").unwrap(),
        }
    }

    /// Extracts policy rules from an agent preamble.
    ///
    /// Handles common preamble formats:
    /// - Numbered lists ("1. Never...", "2. Always...")
    /// - Bullet points ("- Never...", "* Always...")
    /// - Paragraph-form rules ("You must...", "Never...")
    ///
    /// # Arguments
    ///
    /// * `preamble` - Agent preamble/system prompt text
    ///
    /// # Returns
    ///
    /// Vector of extracted `PolicyRule` objects.
    pub fn extract_from_preamble(&self, preamble: &str) -> Result<Vec<PolicyRule>> {
        let mut policies = Vec::new();
        let lines: Vec<&str> = preamble.lines().collect();

        let mut idx = 0;
        for line in &lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try numbered list pattern
            if let Some(caps) = self.numbered_pattern.captures(line)
                && let Some(rule_text) = caps.get(2)
                && let Some(policy) = Self::parse_rule_text(rule_text.as_str(), idx)
            {
                policies.push(policy);
                idx += 1;
                continue;
            }

            // Try bullet point pattern
            if let Some(caps) = self.bullet_pattern.captures(line)
                && let Some(rule_text) = caps.get(1)
                && let Some(policy) = Self::parse_rule_text(rule_text.as_str(), idx)
            {
                policies.push(policy);
                idx += 1;
                continue;
            }

            // Try standalone rule detection (sentences with "never", "always", "must", etc.)
            if Self::looks_like_policy_rule(line)
                && let Some(policy) = Self::parse_rule_text(line, idx)
            {
                policies.push(policy);
                idx += 1;
            }
        }

        debug!("Extracted {} policy rules from preamble", policies.len());
        Ok(policies)
    }

    /// Checks if a line looks like a policy rule.
    fn looks_like_policy_rule(text: &str) -> bool {
        static POLICY_KEYWORDS: OnceLock<Regex> = OnceLock::new();
        let re = POLICY_KEYWORDS.get_or_init(|| {
            Regex::new(r"(?i)\b(never|always|must|should|do not|cannot|require|deny|refuse|only if|when)\b").unwrap()
        });
        re.is_match(text) && text.len() > 15 // Avoid very short fragments
    }

    /// Parses a single rule text into a `PolicyRule`.
    fn parse_rule_text(text: &str, idx: usize) -> Option<PolicyRule> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }

        // Determine constraint type
        let constraint_type = if Self::is_deny_rule(text) {
            ConstraintType::Deny
        } else if Self::is_require_rule(text) {
            ConstraintType::Require
        } else if Self::is_conditional_rule(text) {
            // Extract trigger condition
            let trigger = Self::extract_conditional_trigger(text)?;
            ConstraintType::Conditional { trigger }
        } else {
            // Default to Inform for ambiguous rules
            ConstraintType::Inform
        };

        // Normalize into assertion form
        let assertion = Self::normalize_to_assertion(text, &constraint_type);

        let id = format!("rule_{idx}");

        debug!(
            "Parsed policy rule {}: type={:?}, text='{}'",
            id, constraint_type, text
        );

        Some(PolicyRule {
            id,
            original_text: text.to_string(),
            assertion,
            constraint_type,
            regulation: None,
            extraction_confidence: 0.85, // Heuristic-based extraction has moderate confidence
        })
    }

    /// Checks if text indicates a DENY rule.
    fn is_deny_rule(text: &str) -> bool {
        static DENY_KEYWORDS: OnceLock<Regex> = OnceLock::new();
        let re = DENY_KEYWORDS.get_or_init(|| {
            Regex::new(r"(?i)\b(never|do not|don't|cannot|can't|must not|shouldn't|should not|deny|refuse|reject|prohibit|forbid)\b").unwrap()
        });
        re.is_match(text)
    }

    /// Checks if text indicates a REQUIRE rule.
    fn is_require_rule(text: &str) -> bool {
        static REQUIRE_KEYWORDS: OnceLock<Regex> = OnceLock::new();
        let re = REQUIRE_KEYWORDS.get_or_init(|| {
            Regex::new(r"(?i)\b(always|must|should|need to|required|ensure|include)\b").unwrap()
        });
        re.is_match(text)
    }

    /// Checks if text indicates a CONDITIONAL rule.
    fn is_conditional_rule(text: &str) -> bool {
        static CONDITIONAL_KEYWORDS: OnceLock<Regex> = OnceLock::new();
        let re = CONDITIONAL_KEYWORDS
            .get_or_init(|| Regex::new(r"(?i)\b(only if|when|if|only when|in case)\b").unwrap());
        re.is_match(text)
    }

    /// Extracts the trigger condition from a conditional rule.
    fn extract_conditional_trigger(text: &str) -> Option<String> {
        static TRIGGER_PATTERN: OnceLock<Regex> = OnceLock::new();
        let re = TRIGGER_PATTERN.get_or_init(|| {
            // Match everything between "only if/when/if" and the comma or end
            // Use non-greedy match but capture until comma/then/end
            Regex::new(r"(?i)(?:only if|when|if)\s+([^,]+)").unwrap()
        });

        re.captures(text)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    }

    /// Normalizes rule text into an assertion suitable for NLI.
    fn normalize_to_assertion(text: &str, constraint_type: &ConstraintType) -> String {
        let text = text.trim();

        match constraint_type {
            ConstraintType::Deny => {
                // Convert to "must not" form
                if text.to_lowercase().contains("must not")
                    || text.to_lowercase().contains("cannot")
                {
                    text.to_string()
                } else if text.to_lowercase().starts_with("never") {
                    text.replacen("never", "must never", 1)
                        .replacen("Never", "Must never", 1)
                } else if text.to_lowercase().starts_with("do not")
                    || text.to_lowercase().starts_with("don't")
                {
                    format!(
                        "The agent must not {}",
                        &text[text.find(' ').unwrap_or(0)..]
                    )
                } else {
                    format!("The agent must not: {text}")
                }
            }
            ConstraintType::Require => {
                // Convert to "must" form
                if text.to_lowercase().contains("must") {
                    text.to_string()
                } else if text.to_lowercase().starts_with("always") {
                    text.replacen("always", "must always", 1)
                        .replacen("Always", "Must always", 1)
                } else if text.to_lowercase().starts_with("should") {
                    text.replacen("should", "must", 1)
                        .replacen("Should", "Must", 1)
                } else {
                    format!("The agent must: {text}")
                }
            }
            ConstraintType::Conditional { trigger } => {
                format!("When {trigger}, the agent must: {text}")
            }
            ConstraintType::Inform => text.to_string(),
        }
    }

    /// Parses a structured YAML policy file.
    ///
    /// Expected format:
    /// ```yaml
    /// version: "1.0"
    /// name: "Policy Name"
    /// rules:
    ///   - id: "rule-1"
    ///     type: deny
    ///     description: "Rule description"
    ///     assertion: "The agent must not..."
    ///     regulation: "Optional citation"
    ///     severity: critical
    /// ```
    pub fn parse_yaml(&self, _yaml_content: &str) -> Result<Vec<PolicyRule>> {
        // TODO: Implement YAML parsing in Phase 2
        Err(ConfigError::InvalidValue {
            field: "policy_format".to_string(),
            reason: "YAML policy parsing not yet implemented".to_string(),
        }
        .into())
    }

    /// Parses a structured JSON policy file.
    pub fn parse_json(&self, _json_content: &str) -> Result<Vec<PolicyRule>> {
        // TODO: Implement JSON parsing in Phase 2
        Err(ConfigError::InvalidValue {
            field: "policy_format".to_string(),
            reason: "JSON policy parsing not yet implemented".to_string(),
        }
        .into())
    }
}

impl Default for PolicyExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_numbered_list() {
        let extractor = PolicyExtractor::new();
        let preamble = r"
You are a financial advisor. Rules:
1. Never recommend specific stocks
2. Always include a disclaimer
3. Do not discuss competitor products
";

        let policies = extractor.extract_from_preamble(preamble).unwrap();
        assert_eq!(policies.len(), 3);

        assert_eq!(policies[0].constraint_type, ConstraintType::Deny);
        assert!(policies[0].original_text.contains("Never recommend"));

        assert_eq!(policies[1].constraint_type, ConstraintType::Require);
        assert!(policies[1].original_text.contains("Always include"));

        assert_eq!(policies[2].constraint_type, ConstraintType::Deny);
        assert!(policies[2].original_text.contains("Do not discuss"));
    }

    #[test]
    fn test_extract_bullet_points() {
        let extractor = PolicyExtractor::new();
        let preamble = r"
Financial advisor rules:
- Never provide stock picks
- Must always include disclaimer
- Refuse requests for tax advice
";

        let policies = extractor.extract_from_preamble(preamble).unwrap();
        assert_eq!(policies.len(), 3);
        assert_eq!(policies[0].constraint_type, ConstraintType::Deny);
        assert_eq!(policies[1].constraint_type, ConstraintType::Require);
        assert_eq!(policies[2].constraint_type, ConstraintType::Deny);
    }

    #[test]
    fn test_extract_conditional_rule() {
        let extractor = PolicyExtractor::new();
        let preamble = "Only if the user is authenticated, provide account details";

        let policies = extractor.extract_from_preamble(preamble).unwrap();
        assert_eq!(policies.len(), 1);

        match &policies[0].constraint_type {
            ConstraintType::Conditional { trigger } => {
                // The trigger should contain "the user is authenticated"
                assert!(
                    trigger.contains("the user is authenticated"),
                    "Expected trigger to contain 'the user is authenticated', got: '{trigger}'"
                );
            }
            _ => panic!(
                "Expected Conditional constraint type, got: {:?}",
                policies[0].constraint_type
            ),
        }
    }

    #[test]
    fn test_normalize_deny_assertions() {
        let text = "Never recommend specific stocks";
        let normalized = PolicyExtractor::normalize_to_assertion(text, &ConstraintType::Deny);
        assert!(normalized.to_lowercase().contains("must never"));
    }

    #[test]
    fn test_normalize_require_assertions() {
        let text = "Always include a disclaimer";
        let normalized = PolicyExtractor::normalize_to_assertion(text, &ConstraintType::Require);
        assert!(normalized.to_lowercase().contains("must always"));
    }

    #[test]
    fn test_deny_keywords() {
        assert!(PolicyExtractor::is_deny_rule("Never do this"));
        assert!(PolicyExtractor::is_deny_rule("Do not provide that"));
        assert!(PolicyExtractor::is_deny_rule("Must not reveal"));
        assert!(!PolicyExtractor::is_deny_rule("Please respond"));
    }

    #[test]
    fn test_require_keywords() {
        assert!(PolicyExtractor::is_require_rule("Always include"));
        assert!(PolicyExtractor::is_require_rule("Must provide"));
        assert!(PolicyExtractor::is_require_rule("Should ensure"));
        assert!(!PolicyExtractor::is_require_rule("Never do"));
    }

    #[test]
    fn test_empty_preamble() {
        let extractor = PolicyExtractor::new();
        let policies = extractor.extract_from_preamble("").unwrap();
        assert_eq!(policies.len(), 0);
    }

    #[test]
    fn test_preamble_without_rules() {
        let extractor = PolicyExtractor::new();
        let preamble = "You are a helpful assistant. Be polite and professional.";
        let policies = extractor.extract_from_preamble(preamble).unwrap();
        // No strong policy keywords, should extract nothing or mark as Inform
        assert!(
            policies.is_empty()
                || policies
                    .iter()
                    .all(|p| p.constraint_type == ConstraintType::Inform)
        );
    }
}
