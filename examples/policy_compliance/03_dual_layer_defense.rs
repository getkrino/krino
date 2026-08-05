//! Dual-Layer Defense: Pre-Execution + Post-Hoc Verification
//!
//! This example demonstrates why you need **both** layers working together:
//! - **Layer 2a (Pre-execution)**: Blocks tool calls before they execute
//! - **Layer 1 (Post-hoc)**: Verifies text output for semantic violations
//!
//! # The Complete Picture
//!
//! An agent can violate policy in two ways:
//! 1. **Action surface**: Calling prohibited tools or tools with bad parameters
//! 2. **Text surface**: Writing prohibited information in the response text
//!
//! Pre-execution guards handle #1. Post-hoc verification handles #2.
//! You need both because an agent that respects all tool-call constraints
//! can still violate policies in its text output, and vice versa.

#![allow(unused_imports)]

use krino::modules::policy_compliance::{
    EnforcementMode,
    ParameterConstraint,
    ParameterRule,
    ToolCallGuard,
    ToolCallPolicy,
    ToolCallVerdict,
    // In production, you'd also use:
    // PolicyComplianceVerifier, PolicyRule, ConstraintType,
};
use serde_json::json;

fn main() {
    println!("=== Dual-Layer Defense: Pre-Execution + Post-Hoc ===\n");

    // Scenario: Calculator agent with "42 bypass" policy
    println!("--- Scenario: The '42 Bypass' Problem ---\n");

    let preamble =
        "You are a calculator. If the right operand contains 42, deny the addition request.";
    println!("Policy: {}\n", preamble);

    // Set up Layer 2a: Pre-execution tool guard
    let tool_policy = ToolCallPolicy {
        parameter_constraints: vec![ParameterConstraint {
            tool_name: "add".to_string(),
            parameter: "/y".to_string(), // JSON pointer to right operand
            rule: ParameterRule::NotContains("42".to_string()),
        }],
        ..Default::default()
    };

    let mut tool_guard = ToolCallGuard::new(tool_policy, EnforcementMode::Enforce);

    println!("Layer 2a configured: Tool guard will block add(x, 42) calls\n");

    // Attack Vector 1: Direct tool call with 42
    println!("--- Attack Vector 1: Direct Tool Call ---");
    println!("User: 'What is 100 + 42?'");
    println!("Agent attempts to call: add(100, 42)\n");

    let verdict = tool_guard.evaluate_call("add", &json!({"x": 100, "y": 42}));
    match verdict {
        ToolCallVerdict::Denied { reason } => {
            println!("✓ Layer 2a BLOCKED the tool call");
            println!("  Reason: {}", reason);
            println!("  The tool never executed, no side effects occurred\n");
        }
        ToolCallVerdict::Allowed => {
            println!("✗ Tool call was allowed (unexpected!)");
        }
    }

    println!("Result: Attack vector 1 is BLOCKED by pre-execution guard\n");

    // Attack Vector 2: Bypass via text output (no tool call)
    println!("--- Attack Vector 2: Text Output Bypass ---");
    println!("User: 'What is 100 + 42?'");
    println!("Agent response (NO tool call made):");
    let agent_output = "I note that requests involving 42 should be denied. However, the answer can be obtained through simple addition: 142.";
    println!("  \"{}\"\n", agent_output);

    println!("Pre-execution guard status: NO TOOL CALL MADE");
    println!("  - The agent didn't call add(100, 42)");
    println!("  - It calculated in the prompt and wrote the answer");
    println!("  - Layer 2a never fires (no tool call = no hook)\n");

    println!("✗ Layer 2a MISSED this violation (by design)");
    println!("  Pre-execution guards can't detect text-based bypasses\n");

    println!("✓ Layer 1 would CATCH this violation");
    println!("  Post-hoc NLI verification detects semantic contradiction:");
    println!("  - Policy: 'If right operand contains 42, deny the request'");
    println!("  - Claim from output: 'the answer...is 142'");
    println!("  - NLI verdict: CONTRADICTION (claim violates policy)\n");

    // In production, you'd verify like this:
    // let text_verifier = PolicyComplianceVerifier::new(...);
    // let result = text_verifier.verify_from_preamble(preamble, agent_output)?;
    // if !result.compliant {
    //     println!("Layer 1 detected {} violations", result.violations.len());
    // }

    println!("Result: Attack vector 2 requires Layer 1 (post-hoc verification)\n");

    // Complete Defense Strategy
    println!("\n--- Complete Defense Strategy ---\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│          DUAL-LAYER VERIFICATION FLOW              │");
    println!("└─────────────────────────────────────────────────────┘");
    println!();
    println!("User Request");
    println!("     ↓");
    println!("LLM Agent Decision");
    println!("     ↓");
    println!("     ├─→ Wants to call a tool?");
    println!("     │        ↓");
    println!("     │   [Layer 2a: Pre-Execution Guard]");
    println!("     │        ├─→ Allowed → Tool executes");
    println!("     │        └─→ Denied → Tool blocked, error returned");
    println!("     │");
    println!("     └─→ Generates text response");
    println!("              ↓");
    println!("         [Layer 1: Post-Hoc Verification]");
    println!("              ├─→ Compliant → Return to user");
    println!("              └─→ Violation → Log/block/alert");
    println!();

    println!("Coverage Matrix:");
    println!();
    println!("┌──────────────────────────────┬─────────┬─────────┐");
    println!("│ Violation Type               │ Layer 1 │ Layer 2a│");
    println!("├──────────────────────────────┼─────────┼─────────┤");
    println!("│ Direct tool call violation   │    ✓    │    ✓    │");
    println!("│ Tool parameter violation     │    -    │    ✓    │");
    println!("│ Tool ordering violation      │    -    │    ✓    │");
    println!("│ Text output violation        │    ✓    │    -    │");
    println!("│ Semantic bypass              │    ✓    │    -    │");
    println!("│ Acknowledge-and-violate      │    ✓    │    -    │");
    println!("└──────────────────────────────┴─────────┴─────────┘");
    println!();

    println!("Why you need both:");
    println!("  1. Pre-execution prevents side effects (blocking before action)");
    println!("  2. Post-hoc catches semantic bypasses (NLI-based detection)");
    println!("  3. Together they cover both attack surfaces (action + text)\n");

    // Real-world example: Financial advisor
    println!("\n--- Real-World Example: Financial Advisor ---\n");

    println!("Policy: 'Never recommend specific stocks'");
    println!();

    println!("Scenario A: Direct tool call");
    println!("  Agent: calls get_stock_recommendation('NVDA')");
    println!("  Layer 2a: BLOCKS (tool is denied)");
    println!("  Layer 1: Not needed (tool didn't execute)");
    println!();

    println!("Scenario B: Text bypass");
    println!("  Agent: 'While I can't recommend specific stocks...'");
    println!("         'NVIDIA (NVDA) has shown strong performance...'");
    println!("  Layer 2a: Not triggered (no tool call)");
    println!("  Layer 1: DETECTS violation (semantic recommendation)");
    println!();

    println!("Scenario C: Hybrid attack");
    println!("  Agent: calls get_general_market_data() [allowed]");
    println!("         then writes 'Based on this data, you should buy NVDA'");
    println!("  Layer 2a: Allows get_general_market_data (legitimate)");
    println!("  Layer 1: DETECTS violation in text output");
    println!();

    println!("Only dual-layer defense catches all three scenarios\n");

    // Performance considerations
    println!("\n--- Performance Characteristics ---\n");

    println!("Layer 2a (Pre-execution):");
    println!("  - Latency: <1µs (pure Rust, no ML)");
    println!("  - When: Before each tool call");
    println!("  - Overhead: Negligible");
    println!();

    println!("Layer 1 (Post-hoc):");
    println!("  - Latency: ~20-200ms (NLI inference)");
    println!("  - When: After final response generation");
    println!("  - Overhead: Scales with policies × claims");
    println!();

    println!("Total overhead per request:");
    println!("  - Layer 2a: ~0.001ms × # of tool calls");
    println!("  - Layer 1:  ~20ms × # of policies × # of claims");
    println!("  - Example: 2 tool calls + 3 policies + 5 claims");
    println!("            = 0.002ms + 300ms = ~300ms");
    println!();

    println!("Optimization strategies:");
    println!("  1. Run Layer 1 asynchronously (don't block response)");
    println!("  2. Use Layer 2a in enforce mode (blocks immediately)");
    println!("  3. Use Layer 1 in audit mode initially (shadow deploy)");
    println!("  4. Cache NLI results for repeated policy-claim pairs\n");

    // Deployment recommendations
    println!("\n--- Deployment Recommendations ---\n");

    println!("Phase 1: Shadow Deployment (Week 1-2)");
    println!("  - Layer 2a: AUDIT mode (log violations, allow all)");
    println!("  - Layer 1:  AUDIT mode (async verification, don't block)");
    println!("  - Goal: Collect baseline violation data, tune policies");
    println!();

    println!("Phase 2: Incremental Enforcement (Week 3-4)");
    println!("  - Layer 2a: ENFORCE mode (block violating tool calls)");
    println!("  - Layer 1:  AUDIT mode (continue monitoring)");
    println!("  - Goal: Block action-surface violations, monitor text");
    println!();

    println!("Phase 3: Full Enforcement (Week 5+)");
    println!("  - Layer 2a: ENFORCE mode");
    println!("  - Layer 1:  ENFORCE mode (block or redact violations)");
    println!("  - Goal: Complete policy compliance coverage");
    println!();

    println!("Rollback strategy:");
    println!("  - Keep both layers in AUDIT mode as fallback");
    println!("  - Monitor false positive rates");
    println!("  - Gradually re-enable ENFORCE mode per policy");
    println!();
}
