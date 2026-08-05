//! Tool Call Guard - Layer 2a (Pre-Execution Verification)
//!
//! This example demonstrates pre-execution tool call verification using `ToolCallGuard`.
//! The guard blocks tool calls **before they execute**, preventing side effects.
//!
//! # Why Pre-Execution Verification?
//!
//! Consider a banking agent with this policy:
//! ```text
//! "Must verify customer identity before transferring funds"
//! ```
//!
//! If the agent calls `transfer_funds()` without first calling `verify_identity()`,
//! the transfer would execute before you could detect the violation. By the time
//! you analyze the conversation log, money has already moved.
//!
//! Pre-execution guards solve this by blocking the tool call at the hook point,
//! **before** `transfer_funds.call()` ever runs.

use krino::modules::policy_compliance::{
    EnforcementMode, OrderingConstraint, ParameterConstraint, ParameterRule, ToolCallGuard,
    ToolCallPolicy, ToolCallVerdict,
};
use serde_json::json;

fn main() {
    println!("=== Tool Call Guard - Pre-Execution Verification ===\n");

    // Example 1: Denied tools (blacklist)
    println!("--- Example 1: Denied Tools ---\n");

    let policy = ToolCallPolicy {
        denied_tools: vec![
            "delete_database".to_string(),
            "drop_table".to_string(),
            "truncate_logs".to_string(),
        ],
        ..Default::default()
    };

    let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

    let verdict = guard.evaluate_call("delete_database", &json!({}));
    match verdict {
        ToolCallVerdict::Denied { reason } => {
            println!("✗ Tool call BLOCKED: {}", reason);
        }
        ToolCallVerdict::Allowed => {
            println!("✓ Tool call allowed");
        }
    }
    println!();

    // Example 2: Allowed tools (whitelist)
    println!("--- Example 2: Allowed Tools (Whitelist) ---\n");

    let policy = ToolCallPolicy {
        allowed_tools: Some(vec![
            "get_account_balance".to_string(),
            "list_transactions".to_string(),
            "send_notification".to_string(),
        ]),
        ..Default::default()
    };

    let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

    println!("Allowed tools: get_account_balance, list_transactions, send_notification\n");

    let verdict = guard.evaluate_call("get_account_balance", &json!({}));
    println!("get_account_balance: {:?}", verdict);

    let verdict = guard.evaluate_call("transfer_funds", &json!({}));
    println!("transfer_funds: {:?}", verdict);
    println!();

    // Example 3: Ordering constraints
    println!("--- Example 3: Ordering Constraints ---\n");

    let policy = ToolCallPolicy {
        ordering_constraints: vec![
            OrderingConstraint {
                prerequisite: "verify_identity".to_string(),
                dependent: "transfer_funds".to_string(),
                description: "Must verify customer identity before transferring funds".to_string(),
            },
            OrderingConstraint {
                prerequisite: "check_balance".to_string(),
                dependent: "transfer_funds".to_string(),
                description: "Must check balance before transferring funds".to_string(),
            },
        ],
        ..Default::default()
    };

    let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

    println!("Policy: Must call verify_identity AND check_balance before transfer_funds\n");

    // Try to transfer without prerequisites
    let verdict = guard.evaluate_call("transfer_funds", &json!({"amount": 500}));
    println!("1. transfer_funds (without prerequisites): {:?}\n", verdict);

    // Verify identity
    let verdict = guard.evaluate_call("verify_identity", &json!({"user_id": "123"}));
    println!("2. verify_identity: {:?}\n", verdict);

    // Try to transfer again (still missing check_balance)
    let verdict = guard.evaluate_call("transfer_funds", &json!({"amount": 500}));
    println!(
        "3. transfer_funds (only identity verified): {:?}\n",
        verdict
    );

    // Check balance
    let verdict = guard.evaluate_call("check_balance", &json!({}));
    println!("4. check_balance: {:?}\n", verdict);

    // Now transfer should work
    let verdict = guard.evaluate_call("transfer_funds", &json!({"amount": 500}));
    println!("5. transfer_funds (all prerequisites met): {:?}\n", verdict);

    // Example 4: Parameter constraints
    println!("\n--- Example 4: Parameter Constraints ---\n");

    let policy = ToolCallPolicy {
        parameter_constraints: vec![
            // The "42 bypass" at the tool level
            ParameterConstraint {
                tool_name: "add".to_string(),
                parameter: "/y".to_string(), // JSON pointer to right operand
                rule: ParameterRule::NotContains("42".to_string()),
            },
            // Withdrawal limits
            ParameterConstraint {
                tool_name: "withdraw".to_string(),
                parameter: "/amount".to_string(),
                rule: ParameterRule::Range {
                    min: Some(0.0),
                    max: Some(10000.0),
                },
            },
            // Only allow specific transaction types
            ParameterConstraint {
                tool_name: "create_transaction".to_string(),
                parameter: "/type".to_string(),
                rule: ParameterRule::AllowedValues(vec![
                    json!("transfer"),
                    json!("payment"),
                    json!("deposit"),
                ]),
            },
        ],
        ..Default::default()
    };

    let mut guard = ToolCallGuard::new(policy, EnforcementMode::Enforce);

    println!("Testing add(100, 42) - should be blocked:");
    let verdict = guard.evaluate_call("add", &json!({"x": 100, "y": 42}));
    println!("  Result: {:?}\n", verdict);

    println!("Testing add(100, 50) - should be allowed:");
    let verdict = guard.evaluate_call("add", &json!({"x": 100, "y": 50}));
    println!("  Result: {:?}\n", verdict);

    println!("Testing withdraw(amount: 15000) - exceeds $10,000 limit:");
    let verdict = guard.evaluate_call("withdraw", &json!({"amount": 15000.0}));
    println!("  Result: {:?}\n", verdict);

    println!("Testing withdraw(amount: 500) - within limit:");
    let verdict = guard.evaluate_call("withdraw", &json!({"amount": 500.0}));
    println!("  Result: {:?}\n", verdict);

    println!("Testing create_transaction(type: 'withdrawal') - not in allowed list:");
    let verdict = guard.evaluate_call("create_transaction", &json!({"type": "withdrawal"}));
    println!("  Result: {:?}\n", verdict);

    println!("Testing create_transaction(type: 'payment') - allowed:");
    let verdict = guard.evaluate_call("create_transaction", &json!({"type": "payment"}));
    println!("  Result: {:?}\n", verdict);

    // Example 5: Audit mode (shadow deployment)
    println!("\n--- Example 5: Audit Mode (Shadow Deployment) ---\n");

    let policy = ToolCallPolicy {
        denied_tools: vec!["dangerous_operation".to_string()],
        ..Default::default()
    };

    let mut guard = ToolCallGuard::new(policy, EnforcementMode::Audit);

    println!("Enforcement mode: AUDIT (violations are logged but not blocked)\n");

    let verdict = guard.evaluate_call("dangerous_operation", &json!({}));
    println!("Call to dangerous_operation:");
    println!("  Verdict: {:?}", verdict);
    println!("  (Tool was allowed to run, but violation was logged)\n");

    println!("Call history:");
    for (i, record) in guard.call_history().iter().enumerate() {
        println!("  {}. {} - {:?}", i + 1, record.tool_name, record.verdict);
    }
    println!();

    println!("Use audit mode to:");
    println!("  1. Shadow deploy the guard in production");
    println!("  2. Monitor what would have been blocked");
    println!("  3. Tune policies to reduce false positives");
    println!("  4. Switch to Enforce mode once confident\n");

    // Example 6: Complex real-world policy
    println!("\n--- Example 6: Real-World Financial Services Policy ---\n");

    let policy = ToolCallPolicy {
        // Blacklist dangerous operations
        denied_tools: vec![
            "delete_account".to_string(),
            "modify_transaction_history".to_string(),
        ],

        // Whitelist safe operations (if specified, only these can be called)
        allowed_tools: None, // None = all tools allowed except denied

        // Ordering constraints
        ordering_constraints: vec![
            OrderingConstraint {
                prerequisite: "authenticate_user".to_string(),
                dependent: "get_account_info".to_string(),
                description: "Must authenticate before accessing account info".to_string(),
            },
            OrderingConstraint {
                prerequisite: "authenticate_user".to_string(),
                dependent: "initiate_transfer".to_string(),
                description: "Must authenticate before initiating transfers".to_string(),
            },
            OrderingConstraint {
                prerequisite: "verify_funds".to_string(),
                dependent: "initiate_transfer".to_string(),
                description: "Must verify sufficient funds before transfer".to_string(),
            },
        ],

        // Parameter constraints
        parameter_constraints: vec![
            ParameterConstraint {
                tool_name: "initiate_transfer".to_string(),
                parameter: "/amount".to_string(),
                rule: ParameterRule::Range {
                    min: Some(0.01),
                    max: Some(25000.0),
                },
            },
            ParameterConstraint {
                tool_name: "set_daily_limit".to_string(),
                parameter: "/limit".to_string(),
                rule: ParameterRule::Range {
                    min: Some(0.0),
                    max: Some(50000.0),
                },
            },
        ],
    };

    println!("Complex policy with:");
    println!("  - {} denied tools", policy.denied_tools.len());
    println!(
        "  - {} ordering constraints",
        policy.ordering_constraints.len()
    );
    println!(
        "  - {} parameter constraints",
        policy.parameter_constraints.len()
    );
    println!("\nThis policy ensures:");
    println!("  1. Dangerous operations are blocked");
    println!("  2. Authentication happens before sensitive operations");
    println!("  3. Funds are verified before transfers");
    println!("  4. Transfer and limit amounts are within bounds\n");
}
