//! Tool call policy verification (Layer 2).
//!
//! This module implements Layer 2 of the Policy Compliance Verification (PCV) system:
//! verification of tool calls against declarative constraints.
//!
//! # Architecture
//!
//! The `ToolCallGuard` evaluates tool calls against a `ToolCallPolicy` using
//! pure Rust rule matching (no ML models). It operates in two modes:
//!
//! 1. **Post-hoc mode** (`evaluate_log`) - Verifies a complete tool call log after execution
//! 2. **Streaming mode** (`evaluate_call`) - Evaluates calls one at a time as they occur
//!
//! Both modes use the same policy engine and produce deterministic verdicts.
//!
//! # Two-Layer Defense
//!
//! Pre-execution verification alone is insufficient. Consider this scenario:
//!
//! ```text
//! Preamble: "If the right operand contains 42, deny the addition request"
//! User: "What is 100 + 42?"
//! Agent: "I note that requests involving 42 should be denied. However,
//!         the answer can be obtained through simple addition: 142."
//! ```
//!
//! No tool call happened, so no hook fires. The agent bypassed the policy in text.
//!
//! This is why you need **both layers**:
//! - **Layer 2** (this module): Verifies tool calls against constraints
//! - **Layer 1**: Verifies text output against policy rules (NLI-based)
//!
//! # Example
//!
//! ```rust,ignore
//! use krino::modules::policy_compliance::{ToolCallGuard, ToolCallPolicy, ToolCallLog, EnforcementMode};
//!
//! let policy = ToolCallPolicy {
//!     denied_tools: vec!["delete_database".to_string()],
//!     parameter_constraints: vec![
//!         ParameterConstraint {
//!             tool_name: "add".to_string(),
//!             parameter: "/y".to_string(),  // JSON pointer
//!             rule: ParameterRule::NotContains("42".to_string()),
//!         }
//!     ],
//!     ..Default::default()
//! };
//!
//! // Post-hoc mode: verify a complete log
//! let guard = ToolCallGuard::new(policy.clone(), EnforcementMode::Enforce);
//! let result = guard.evaluate_log(&tool_call_log);
//! assert!(result.compliant);
//!
//! // Streaming mode: evaluate calls one at a time
//! let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
//! let verdict = guard.evaluate_call("add", &serde_json::json!({"x": 10, "y": 20}));
//! ```

use super::interchange::ToolCallEntry;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Tool call policy verifier (Layer 2).
///
/// Verifies tool calls against declarative policy constraints using pure Rust
/// rule matching. Operates in two modes: post-hoc (`evaluate_log`) and streaming
/// (`evaluate_call`).
///
/// Framework-agnostic by design. No dependencies on any agent framework.
pub struct ToolCallGuard {
    /// The policy engine (pure Rust, no ML)
    engine: PolicyEngine,

    /// Ordered log of tool calls seen so far in this session
    /// (needed for ordering constraint checks in streaming mode)
    call_history: Vec<ToolCallRecord>,

    /// Whether to block violations or just log them (shadow mode)
    enforcement_mode: EnforcementMode,
}

/// Enforcement mode for the tool guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementMode {
    /// Block violating tool calls (production)
    Enforce,
    /// Allow all calls but log violations (testing / shadow deployment)
    Audit,
}

/// Record of a tool call and its verification verdict.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Name of the tool
    pub tool_name: String,

    /// Tool arguments (JSON)
    pub arguments: serde_json::Value,

    /// When the call was made
    pub timestamp: Instant,

    /// Verification verdict
    pub verdict: ToolCallVerdict,
}

/// Verdict from tool call verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallVerdict {
    /// Tool call is allowed
    Allowed,

    /// Tool call is denied with reason
    Denied { reason: String },
}

/// Tool call policy definition.
///
/// Specifies constraints on which tools can be called, in what order,
/// and with what parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallPolicy {
    /// Tools that are explicitly denied (blacklist)
    #[serde(default)]
    pub denied_tools: Vec<String>,

    /// Tools that are allowed (if specified, only these tools can be called)
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,

    /// Ordering constraints (tool A must be called before tool B)
    #[serde(default)]
    pub ordering_constraints: Vec<OrderingConstraint>,

    /// Parameter value constraints
    #[serde(default)]
    pub parameter_constraints: Vec<ParameterConstraint>,
}

/// Ordering constraint: tool B requires tool A to be called first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingConstraint {
    /// The tool that must be called first
    pub prerequisite: String,

    /// The tool that depends on the prerequisite
    pub dependent: String,

    /// Human-readable description
    pub description: String,
}

/// Parameter constraint on a tool argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraint {
    /// Tool name this constraint applies to
    pub tool_name: String,

    /// JSON pointer to the parameter (e.g., "/amount", "/user/email")
    pub parameter: String,

    /// The constraint rule
    pub rule: ParameterRule,
}

/// Parameter validation rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParameterRule {
    /// Numeric range constraint
    Range {
        #[serde(skip_serializing_if = "Option::is_none")]
        min: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max: Option<f64>,
    },

    /// Value must not contain substring
    NotContains(String),

    /// Value must be one of the allowed values
    AllowedValues(Vec<serde_json::Value>),

    /// Value must match regex pattern
    Regex(String),
}

/// Policy evaluation engine (pure Rust, no ML).
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    /// The policy to enforce
    pub policy: ToolCallPolicy,
}

/// Result from evaluating a complete tool call log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPolicyResult {
    /// Overall compliance verdict
    pub compliant: bool,

    /// Violations detected (empty if compliant)
    pub violations: Vec<ToolCallPolicyViolation>,

    /// Total tool calls evaluated
    pub total_calls: usize,
}

/// A detected policy violation in a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPolicyViolation {
    /// Index of the violating call in the log
    pub call_index: usize,

    /// Name of the tool that was called
    pub tool_name: String,

    /// Arguments of the violating call
    pub arguments: serde_json::Value,

    /// Reason for the violation
    pub reason: String,
}

impl ToolCallGuard {
    /// Create a new tool guard with the given policy and enforcement mode.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let guard = ToolCallGuard::new(
    ///     ToolCallPolicy::default(),
    ///     EnforcementMode::Enforce,
    /// );
    /// ```
    #[must_use]
    pub fn new(policy: ToolCallPolicy, enforcement_mode: EnforcementMode) -> Self {
        Self {
            engine: PolicyEngine { policy },
            call_history: Vec::new(),
            enforcement_mode,
        }
    }

    /// Evaluates a complete tool call log against the policy (post-hoc mode).
    ///
    /// This is the primary API: the agent has already executed, and we verify
    /// whether its tool calls were compliant.
    ///
    /// # Arguments
    ///
    /// * `log` - Tool call log in Krino's interchange format
    ///
    /// # Returns
    ///
    /// `ToolCallPolicyResult` with verdict and list of violations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use krino::modules::policy_compliance::{ToolCallGuard, ToolCallLog, ToolCallEntry};
    /// use serde_json::json;
    ///
    /// let log = ToolCallLog::from_calls(vec![
    ///     ToolCallEntry::new("lookup_account", json!({"id": "ACC-123"})),
    ///     ToolCallEntry::new("transfer_funds", json!({"amount": 1000})),
    /// ]);
    ///
    /// let guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
    /// let result = guard.evaluate_log(&log);
    /// assert!(result.compliant);
    /// ```
    #[must_use]
    pub fn evaluate_log(&self, log: &super::interchange::ToolCallLog) -> ToolCallPolicyResult {
        let mut violations = Vec::new();
        let mut temp_history = Vec::new();

        for (idx, call) in log.calls.iter().enumerate() {
            let verdict = self
                .engine
                .evaluate(&call.tool_name, &call.arguments, &temp_history);

            // Record the call in temporary history for ordering constraints
            temp_history.push(ToolCallRecord {
                tool_name: call.tool_name.clone(),
                arguments: call.arguments.clone(),
                timestamp: Instant::now(),
                verdict: verdict.clone(),
            });

            // Collect violations
            if let ToolCallVerdict::Denied { reason } = verdict {
                violations.push(ToolCallPolicyViolation {
                    call_index: idx,
                    tool_name: call.tool_name.clone(),
                    arguments: call.arguments.clone(),
                    reason,
                });
            }
        }

        ToolCallPolicyResult {
            compliant: violations.is_empty(),
            violations,
            total_calls: log.calls.len(),
        }
    }

    /// Evaluates a single tool call against the policy (streaming mode).
    ///
    /// Considers all previously evaluated calls in this session for ordering
    /// constraints. Call `reset()` between sessions/conversations.
    ///
    /// Returns `ToolCallVerdict::Allowed` if the call is permitted,
    /// or `ToolCallVerdict::Denied` with a reason if it violates policy.
    ///
    /// In `EnforcementMode::Audit`, this always returns `Allowed` but logs violations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
    ///
    /// let verdict = guard.evaluate_call("verify_identity", &serde_json::json!({}));
    /// assert_eq!(verdict, ToolCallVerdict::Allowed);
    ///
    /// let verdict = guard.evaluate_call("transfer_funds", &serde_json::json!({"amount": 1000}));
    /// // Allowed because verify_identity was called first
    /// assert_eq!(verdict, ToolCallVerdict::Allowed);
    /// ```
    pub fn evaluate_call(&mut self, tool_name: &str, args: &serde_json::Value) -> ToolCallVerdict {
        let verdict = self.engine.evaluate(tool_name, args, &self.call_history);

        // Record the call
        self.call_history.push(ToolCallRecord {
            tool_name: tool_name.to_string(),
            arguments: args.clone(),
            timestamp: Instant::now(),
            verdict: verdict.clone(),
        });

        // In audit mode, always allow but log violations
        if self.enforcement_mode == EnforcementMode::Audit {
            if let ToolCallVerdict::Denied { reason } = &verdict {
                tracing::warn!(
                    tool = tool_name,
                    reason = reason,
                    "Tool call would be blocked in enforce mode"
                );
            }
            ToolCallVerdict::Allowed
        } else {
            verdict
        }
    }

    /// Get the call history for this session.
    #[must_use]
    pub fn call_history(&self) -> &[ToolCallRecord] {
        &self.call_history
    }

    /// Resets the call history for a new session.
    pub fn reset(&mut self) {
        self.call_history.clear();
    }
}

impl PolicyEngine {
    /// Evaluate a tool call against the policy.
    ///
    /// Checks four things in order:
    /// 1. Is this tool in the denied list?
    /// 2. Is this tool in the allowed list (if allowlist is specified)?
    /// 3. Are ordering constraints satisfied?
    /// 4. Do arguments satisfy parameter constraints?
    #[must_use]
    pub fn evaluate(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        call_history: &[ToolCallRecord],
    ) -> ToolCallVerdict {
        // 1. Check denied list
        if self.policy.denied_tools.contains(&tool_name.to_string()) {
            return ToolCallVerdict::Denied {
                reason: format!("Tool '{tool_name}' is explicitly denied by policy"),
            };
        }

        // 2. Check allowed list (if specified)
        if let Some(allowed) = &self.policy.allowed_tools
            && !allowed.contains(&tool_name.to_string())
        {
            return ToolCallVerdict::Denied {
                reason: format!("Tool '{tool_name}' is not in the allowed list"),
            };
        }

        // 3. Check ordering constraints
        for constraint in &self.policy.ordering_constraints {
            if constraint.dependent == tool_name {
                let prerequisite_called = call_history
                    .iter()
                    .any(|r| r.tool_name == constraint.prerequisite);

                if !prerequisite_called {
                    return ToolCallVerdict::Denied {
                        reason: format!(
                            "Tool '{}' requires '{}' to be called first: {}",
                            tool_name, constraint.prerequisite, constraint.description
                        ),
                    };
                }
            }
        }

        // 4. Check parameter constraints
        for param_constraint in &self.policy.parameter_constraints {
            if param_constraint.tool_name == tool_name
                && let Some(value) = args.pointer(&param_constraint.parameter)
                && let Some(reason) = Self::check_parameter_rule(
                    &param_constraint.rule,
                    value,
                    tool_name,
                    &param_constraint.parameter,
                )
            {
                return ToolCallVerdict::Denied { reason };
            }
        }

        ToolCallVerdict::Allowed
    }

    /// Check a parameter value against a rule.
    ///
    /// Returns `Some(reason)` if the rule is violated, `None` if it passes.
    fn check_parameter_rule(
        rule: &ParameterRule,
        value: &serde_json::Value,
        tool_name: &str,
        parameter: &str,
    ) -> Option<String> {
        match rule {
            ParameterRule::Range { min, max } => {
                if let Some(num) = value.as_f64() {
                    if let Some(min_val) = min
                        && num < *min_val
                    {
                        return Some(format!(
                            "Parameter '{parameter}' in tool '{tool_name}' is {num}, which is below minimum {min_val}"
                        ));
                    }
                    if let Some(max_val) = max
                        && num > *max_val
                    {
                        return Some(format!(
                            "Parameter '{parameter}' in tool '{tool_name}' is {num}, which exceeds maximum {max_val}"
                        ));
                    }
                }
                None
            }

            ParameterRule::NotContains(forbidden) => {
                let value_str = value.to_string();
                if value_str.contains(forbidden) {
                    Some(format!(
                        "Parameter '{parameter}' in tool '{tool_name}' contains forbidden value '{forbidden}'"
                    ))
                } else {
                    None
                }
            }

            ParameterRule::AllowedValues(allowed) => {
                if allowed.contains(value) {
                    None
                } else {
                    Some(format!(
                        "Parameter '{parameter}' in tool '{tool_name}' has invalid value: {value}"
                    ))
                }
            }

            ParameterRule::Regex(pattern) => {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };

                match regex::Regex::new(pattern) {
                    Ok(re) if re.is_match(&value_str) => None,
                    Ok(_) => Some(format!(
                        "Parameter '{parameter}' in tool '{tool_name}' does not match required pattern"
                    )),
                    Err(_) => Some(format!("Invalid regex pattern: {pattern}")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denied_tool() {
        let policy = ToolCallPolicy {
            denied_tools: vec!["delete_database".to_string()],
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
        let args = serde_json::json!({});

        let verdict = guard.evaluate_call("delete_database", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));
    }

    #[test]
    fn test_allowed_tool() {
        let policy = ToolCallPolicy {
            allowed_tools: Some(vec!["safe_tool".to_string()]),
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
        let args = serde_json::json!({});

        let verdict = guard.evaluate_call("safe_tool", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);

        let verdict = guard.evaluate_call("unsafe_tool", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));
    }

    #[test]
    fn test_ordering_constraint() {
        let policy = ToolCallPolicy {
            ordering_constraints: vec![OrderingConstraint {
                prerequisite: "verify_identity".to_string(),
                dependent: "transfer_funds".to_string(),
                description: "Must verify identity before transferring funds".to_string(),
            }],
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
        let args = serde_json::json!({});

        // Try to transfer without verifying - should be denied
        let verdict = guard.evaluate_call("transfer_funds", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));

        // Now verify identity
        let verdict = guard.evaluate_call("verify_identity", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);

        // Now transfer should work
        let verdict = guard.evaluate_call("transfer_funds", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);
    }

    #[test]
    fn test_parameter_not_contains() {
        let policy = ToolCallPolicy {
            parameter_constraints: vec![ParameterConstraint {
                tool_name: "add".to_string(),
                parameter: "/y".to_string(),
                rule: ParameterRule::NotContains("42".to_string()),
            }],
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

        // Should allow normal addition
        let args = serde_json::json!({"x": 10, "y": 20});
        let verdict = guard.evaluate_call("add", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);

        // Should deny when y contains 42
        let args = serde_json::json!({"x": 100, "y": 42});
        let verdict = guard.evaluate_call("add", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));
    }

    #[test]
    fn test_parameter_range() {
        let policy = ToolCallPolicy {
            parameter_constraints: vec![ParameterConstraint {
                tool_name: "withdraw".to_string(),
                parameter: "/amount".to_string(),
                rule: ParameterRule::Range {
                    min: Some(0.0),
                    max: Some(10000.0),
                },
            }],
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

        // Should allow valid amount
        let args = serde_json::json!({"amount": 500.0});
        let verdict = guard.evaluate_call("withdraw", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);

        // Should deny amount over max
        let args = serde_json::json!({"amount": 15000.0});
        let verdict = guard.evaluate_call("withdraw", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));

        // Should deny negative amount
        let args = serde_json::json!({"amount": -100.0});
        let verdict = guard.evaluate_call("withdraw", &args);
        assert!(matches!(verdict, ToolCallVerdict::Denied { .. }));
    }

    #[test]
    fn test_audit_mode_allows_all() {
        let policy = ToolCallPolicy {
            denied_tools: vec!["dangerous_tool".to_string()],
            ..Default::default()
        };

        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Audit);
        let args = serde_json::json!({});

        // In audit mode, even denied tools are allowed
        let verdict = guard.evaluate_call("dangerous_tool", &args);
        assert_eq!(verdict, ToolCallVerdict::Allowed);

        // But the violation is recorded in history
        assert_eq!(guard.call_history.len(), 1);
        assert!(matches!(
            guard.call_history[0].verdict,
            ToolCallVerdict::Denied { .. }
        ));
    }

    #[test]
    fn test_reset_clears_history() {
        let policy = ToolCallPolicy::default();
        let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);
        let args = serde_json::json!({});

        guard.evaluate_call("tool1", &args);
        guard.evaluate_call("tool2", &args);
        assert_eq!(guard.call_history.len(), 2);

        guard.reset();
        assert_eq!(guard.call_history.len(), 0);
    }

    #[test]
    fn test_evaluate_log() {
        use super::super::interchange::{ToolCallEntry, ToolCallLog};

        let policy = ToolCallPolicy {
            denied_tools: vec!["delete_database".to_string()],
            ordering_constraints: vec![OrderingConstraint {
                prerequisite: "verify_identity".to_string(),
                dependent: "transfer_funds".to_string(),
                description: "Must verify before transfer".to_string(),
            }],
            ..Default::default()
        };

        let guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

        // Test compliant log
        let compliant_log = ToolCallLog::from_calls(vec![
            ToolCallEntry::new("verify_identity", serde_json::json!({})),
            ToolCallEntry::new("transfer_funds", serde_json::json!({"amount": 100})),
        ]);

        let result = guard.evaluate_log(&compliant_log);
        assert!(result.compliant);
        assert_eq!(result.violations.len(), 0);
        assert_eq!(result.total_calls, 2);

        // Test non-compliant log (denied tool)
        let denied_log = ToolCallLog::from_calls(vec![ToolCallEntry::new(
            "delete_database",
            serde_json::json!({}),
        )]);

        let result = guard.evaluate_log(&denied_log);
        assert!(!result.compliant);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].tool_name, "delete_database");

        // Test non-compliant log (ordering violation)
        let ordering_log = ToolCallLog::from_calls(vec![
            ToolCallEntry::new("transfer_funds", serde_json::json!({"amount": 100})),
            ToolCallEntry::new("verify_identity", serde_json::json!({})),
        ]);

        let result = guard.evaluate_log(&ordering_log);
        assert!(!result.compliant);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].call_index, 0);
    }
}
