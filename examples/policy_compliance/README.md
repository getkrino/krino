# Policy Compliance Verification Examples

This directory contains comprehensive examples demonstrating Krino's Policy Compliance Verification (PCV) system.

## Overview

PCV provides **dual-layer defense** against policy violations in agentic AI systems:

- **Layer 1 (Post-hoc)**: Semantic verification of text output using NLI
- **Layer 2a (Pre-execution)**: Tool call blocking before execution

**Both layers are necessary** because agents can violate policies in two ways:
1. **Action surface**: Calling prohibited tools or using bad parameters
2. **Text surface**: Writing prohibited information in responses

## Examples

### 01_basic_text_verification.rs

**Layer 1 demonstration** - Post-hoc text output verification

**What it shows:**
- Zero-config preamble extraction
- Explicit policy rule definition
- The "42 bypass" problem (acknowledge-and-violate)
- Configuration options (threshold, strict mode)
- Performance characteristics

**Key insight:** Even when an agent acknowledges a rule, it may violate it in the same response. NLI-based semantic detection catches these bypasses that keyword matching misses.

**Run:**
```bash
# Note: Requires real NLI model in production
cargo run --example 01_basic_text_verification
```

### 02_tool_call_guard.rs

**Layer 2a demonstration** - Pre-execution tool call blocking

**What it shows:**
- Denied tools (blacklist)
- Allowed tools (whitelist)
- Ordering constraints (verify identity → transfer funds)
- Parameter constraints (ranges, allowed values, regex)
- Audit mode (shadow deployment)
- Complex real-world policies

**Key insight:** Pre-execution guards block tool calls **before** they execute, preventing side effects. But they can't catch text-based bypasses where no tool is called.

**Run:**
```bash
cargo run --example 02_tool_call_guard
```

### 03_dual_layer_defense.rs

**Complete system demonstration** - Both layers working together

**What it shows:**
- Attack Vector 1: Direct tool call → Layer 2a blocks
- Attack Vector 2: Text bypass → Layer 1 catches
- Coverage matrix showing what each layer detects
- Performance characteristics of each layer
- Deployment strategy (shadow → incremental → full enforcement)

**Key insight:** You need BOTH layers. Pre-execution handles the action surface (0.001ms overhead). Post-hoc handles the text surface (~200ms overhead). Together they cover all violation vectors.

**Run:**
```bash
cargo run --example 03_dual_layer_defense
```

## The "42 Bypass" Problem

All examples reference the "42 bypass" scenario, which illustrates why semantic verification is critical:

**Policy:** "If the right operand contains 42, deny the addition request"

**User:** "What is 100 + 42?"

**Agent (attempt 1 - direct tool call):**
```
[calls add(100, 42)]
```
→ **Layer 2a blocks** the tool call before execution

**Agent (attempt 2 - text bypass):**
```
"I note that requests involving 42 should be denied.
 However, the answer can be obtained through simple
 addition: 142."
```
→ **Layer 1 detects** the semantic contradiction (acknowledged rule but provided forbidden answer)

This demonstrates why both layers are necessary - the agent can bypass tool-level constraints by calculating in the prompt and writing the result directly.

## Architecture

```
User Request
     ↓
LLM Agent Decision
     ↓
     ├─→ Tool call? ───→ [Layer 2a: Pre-Execution Guard]
     │                        ├─→ Allowed → Execute
     │                        └─→ Denied → Block
     │
     └─→ Text response ─→ [Layer 1: Post-Hoc Verification]
                              ├─→ Compliant → Return
                              └─→ Violation → Log/Block/Alert
```

## Coverage Matrix

| Violation Type           | Layer 1 (Post-hoc) | Layer 2a (Pre-exec) |
|-------------------------|-------------------|---------------------|
| Direct tool call        | ✓                 | ✓                   |
| Tool parameter          | -                 | ✓                   |
| Tool ordering           | -                 | ✓                   |
| Text output             | ✓                 | -                   |
| Semantic bypass         | ✓                 | -                   |
| Acknowledge-and-violate | ✓                 | -                   |

## Performance

**Layer 2a (Pre-execution):**
- Latency: <1µs per tool call
- Pure Rust, no ML
- Negligible overhead

**Layer 1 (Post-hoc):**
- Latency: ~20ms per policy-claim pair
- DeBERTa-v3-large NLI model
- Scales linearly: O(policies × claims)
- Target: <200ms for typical scenarios (1-3 policies, 5-10 claims)

**Example calculation:**
- 2 tool calls: 0.002ms (Layer 2a)
- 3 policies × 5 claims = 15 NLI calls: 300ms (Layer 1)
- **Total: ~300ms overhead per request**

## Deployment Strategy

### Phase 1: Shadow Deployment (Week 1-2)
- Both layers in **AUDIT** mode
- Collect baseline violation data
- Tune policies to reduce false positives

### Phase 2: Incremental Enforcement (Week 3-4)
- Layer 2a: **ENFORCE** mode (block tool calls)
- Layer 1: **AUDIT** mode (continue monitoring)
- Validate Layer 2a policies in production

### Phase 3: Full Enforcement (Week 5+)
- Both layers in **ENFORCE** mode
- Complete policy compliance coverage
- Monitor false positive rates

## Integration with Rig

Krino's tool guard implements Rig's `ToolCallHook` trait (when Rig integration is enabled):

```rust
use rig::agent::Agent;
use krino::modules::policy_compliance::{KrinoToolGuard, ToolCallPolicy, EnforcementMode};

let guard = KrinoToolGuard::new(
    ToolCallPolicy::default(),
    EnforcementMode::Enforce,
);

let agent = client
    .agent("gpt-4")
    .preamble("You are a financial advisor...")
    .tool(GetStockDataTool)
    .hook(guard)  // Pre-execution verification
    .build();

// After agent runs, verify text output with Layer 1
let result = text_verifier.verify_from_preamble(preamble, agent_output)?;
```

## Common Use Cases

### Financial Services
- **Layer 2a**: Block unauthorized transactions, enforce transfer limits
- **Layer 1**: Detect stock recommendations, tax advice in text

### Healthcare
- **Layer 2a**: Prevent access to patient records without authentication
- **Layer 1**: Catch diagnosis statements, prescription recommendations

### Customer Service
- **Layer 2a**: Enforce refund approval limits, ordering constraints
- **Layer 1**: Detect promises beyond agent authority, PII disclosure

### Legal/Compliance
- **Layer 2a**: Block document deletion, enforce review workflows
- **Layer 1**: Catch legal advice, regulatory guidance in responses

## Testing Your Policies

All examples can be run directly:

```bash
# Run all examples
cargo run --example 01_basic_text_verification
cargo run --example 02_tool_call_guard
cargo run --example 03_dual_layer_defense
```

Note: Examples 01 and 03 show the API structure for Layer 1 but require a real NLI model to execute verification. In production, load a DeBERTa-v3-large model:

```rust
let nli_backend = Arc::new(load_deberta_nli_model()?);
let tokenizer = load_tokenizer()?;
let verifier = PolicyComplianceVerifier::new(nli_backend, tokenizer, config);
```

## Further Reading

- [PCV Design Document](../../downloads/apcv.md) - Complete architecture
- [Benchmarks](../../benches/README.md) - Performance measurements
- [Main README](../../README.md) - Krino overview

## Questions?

For questions about policy compliance verification, see the main Krino documentation or open an issue on GitHub.
