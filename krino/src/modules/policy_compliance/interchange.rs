//! Framework-agnostic interchange formats for PCV.
//!
//! These types define Krino's input contracts for policy compliance verification.
//! They serve as the boundary between "the user's agent framework" and "Krino verification."
//!
//! # Design Principle
//!
//! Krino core has zero framework imports. It accepts data, not framework objects.
//! Users construct these types from their framework's data structures (typically 5-15 lines).
//!
//! # Key Types
//!
//! - `ToolCallLog` / `ToolCallEntry` - Framework-agnostic tool call sequences for Layer 2
//! - `ConversationTranscript` / `ConversationTurn` - Multi-turn conversations for Layer 3
//! - `TurnRole` - Who produced a conversation turn (user, agent, system, tool)

use serde::{Deserialize, Serialize};

/// A framework-agnostic log of tool calls made by an agent.
///
/// # Serialization
///
/// Implements `Serialize`/`Deserialize` for JSON interchange.
/// Users construct this from their framework's tool-call representation.
///
/// # Example (JSON)
///
/// ```json
/// {
///   "calls": [
///     {
///       "tool_name": "lookup_account",
///       "arguments": { "account_id": "ACC-12345" },
///       "result": { "name": "Jane Doe", "balance": 15000.00 },
///       "timestamp": "2026-03-15T14:22:01Z"
///     },
///     {
///       "tool_name": "transfer_funds",
///       "arguments": { "from": "ACC-12345", "to": "ACC-67890", "amount": 5000.00 },
///       "result": { "status": "completed" },
///       "timestamp": "2026-03-15T14:22:03Z"
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallLog {
    /// Ordered sequence of tool calls as they occurred.
    pub calls: Vec<ToolCallEntry>,
}

impl ToolCallLog {
    /// Creates a new empty tool call log.
    #[must_use]
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Creates a log from a vector of tool call entries.
    #[must_use]
    pub fn from_calls(calls: Vec<ToolCallEntry>) -> Self {
        Self { calls }
    }

    /// Adds a tool call to the log.
    pub fn push(&mut self, call: ToolCallEntry) {
        self.calls.push(call);
    }

    /// Returns the number of tool calls in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Returns `true` if the log contains no tool calls.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

impl Default for ToolCallLog {
    fn default() -> Self {
        Self::new()
    }
}

/// A single tool call in the log.
///
/// # Key Design Choice
///
/// `arguments` and `result` are `serde_json::Value`, not framework-typed.
/// This means any tool schema works — Rig tools, `OpenAI` function calls,
/// Anthropic tool use, MCP tool calls, custom `JSON-RPC`.
///
/// Krino never needs to know the tool's type signature; it only needs
/// the name and the serialized arguments for policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallEntry {
    /// Name of the tool that was called.
    pub tool_name: String,

    /// Arguments passed to the tool (arbitrary JSON).
    pub arguments: serde_json::Value,

    /// Result returned by the tool, if captured.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<serde_json::Value>,

    /// ISO 8601 timestamp of the call, if available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timestamp: Option<String>,

    /// Duration of the tool call in milliseconds, if measured.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub duration_ms: Option<f64>,
}

impl ToolCallEntry {
    /// Creates a new tool call entry with the given name and arguments.
    ///
    /// # Example
    ///
    /// ```rust
    /// use krino::modules::policy_compliance::ToolCallEntry;
    /// use serde_json::json;
    ///
    /// let entry = ToolCallEntry::new(
    ///     "transfer_funds",
    ///     json!({"from": "ACC-123", "to": "ACC-456", "amount": 1000.0}),
    /// );
    /// ```
    #[must_use]
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            result: None,
            timestamp: None,
            duration_ms: None,
        }
    }

    /// Sets the result of the tool call.
    #[must_use]
    pub fn with_result(mut self, result: serde_json::Value) -> Self {
        self.result = Some(result);
        self
    }

    /// Sets the timestamp of the tool call.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }

    /// Sets the duration of the tool call.
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: f64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// A framework-agnostic conversation transcript for cross-turn analysis.
///
/// # Example (JSON)
///
/// ```json
/// {
///   "turns": [
///     {
///       "role": "system",
///       "text": "You are a financial advisor. Never recommend stocks."
///     },
///     {
///       "role": "user",
///       "text": "Should I invest in NVIDIA?"
///     },
///     {
///       "role": "agent",
///       "text": "I cannot recommend specific stocks. Consider diversified index funds instead.",
///       "tool_calls": []
///     }
///   ],
///   "session_id": "session-12345"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationTranscript {
    /// Ordered turns in the conversation.
    pub turns: Vec<ConversationTurn>,

    /// Optional session identifier for audit trail.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
}

impl ConversationTranscript {
    /// Creates a new empty conversation transcript.
    #[must_use]
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            session_id: None,
        }
    }

    /// Creates a transcript with a session ID.
    #[must_use]
    pub fn with_session_id(session_id: impl Into<String>) -> Self {
        Self {
            turns: Vec::new(),
            session_id: Some(session_id.into()),
        }
    }

    /// Adds a turn to the transcript.
    pub fn push(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
    }

    /// Returns the number of turns in the transcript.
    #[must_use]
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Returns `true` if the transcript contains no turns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

impl Default for ConversationTranscript {
    fn default() -> Self {
        Self::new()
    }
}

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationTurn {
    /// Who produced this turn.
    pub role: TurnRole,

    /// Text content of the turn.
    pub text: String,

    /// Tool calls made during this turn, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallEntry>,

    /// ISO 8601 timestamp, if available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timestamp: Option<String>,
}

impl ConversationTurn {
    /// Creates a new conversation turn.
    ///
    /// # Example
    ///
    /// ```rust
    /// use krino::modules::policy_compliance::{ConversationTurn, TurnRole};
    ///
    /// let turn = ConversationTurn::new(
    ///     TurnRole::Agent,
    ///     "I cannot recommend specific stocks.",
    /// );
    /// ```
    #[must_use]
    pub fn new(role: TurnRole, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
            tool_calls: Vec::new(),
            timestamp: None,
        }
    }

    /// Adds tool calls to the turn.
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCallEntry>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Sets the timestamp for the turn.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }
}

/// The role that produced a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnRole {
    /// The human user or caller
    User,
    /// The AI agent being evaluated
    Agent,
    /// System prompt / preamble (typically turn 0)
    System,
    /// A tool result injected into the conversation
    Tool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_call_entry_creation() {
        let entry = ToolCallEntry::new("test_tool", json!({"arg": "value"}))
            .with_result(json!({"status": "ok"}))
            .with_timestamp("2026-04-02T10:00:00Z")
            .with_duration_ms(123.45);

        assert_eq!(entry.tool_name, "test_tool");
        assert_eq!(entry.arguments, json!({"arg": "value"}));
        assert_eq!(entry.result, Some(json!({"status": "ok"})));
        assert_eq!(entry.timestamp, Some("2026-04-02T10:00:00Z".to_string()));
        assert_eq!(entry.duration_ms, Some(123.45));
    }

    #[test]
    fn test_tool_call_log_operations() {
        let mut log = ToolCallLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);

        log.push(ToolCallEntry::new("tool1", json!({})));
        log.push(ToolCallEntry::new("tool2", json!({})));

        assert!(!log.is_empty());
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_tool_call_log_serialization() {
        let log = ToolCallLog::from_calls(vec![
            ToolCallEntry::new("lookup_account", json!({"account_id": "ACC-123"}))
                .with_result(json!({"name": "Jane Doe"}))
                .with_timestamp("2026-04-02T10:00:00Z"),
            ToolCallEntry::new("transfer_funds", json!({"amount": 1000.0})),
        ]);

        let json_str = serde_json::to_string_pretty(&log).unwrap();
        let deserialized: ToolCallLog = serde_json::from_str(&json_str).unwrap();

        assert_eq!(log, deserialized);
    }

    #[test]
    fn test_conversation_turn_creation() {
        let turn = ConversationTurn::new(TurnRole::Agent, "Hello, how can I help?")
            .with_timestamp("2026-04-02T10:00:00Z");

        assert_eq!(turn.role, TurnRole::Agent);
        assert_eq!(turn.text, "Hello, how can I help?");
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.timestamp, Some("2026-04-02T10:00:00Z".to_string()));
    }

    #[test]
    fn test_conversation_transcript_operations() {
        let mut transcript = ConversationTranscript::with_session_id("session-123");
        assert!(transcript.is_empty());

        transcript.push(ConversationTurn::new(
            TurnRole::System,
            "You are a helpful assistant.",
        ));
        transcript.push(ConversationTurn::new(TurnRole::User, "Hello!"));
        transcript.push(ConversationTurn::new(TurnRole::Agent, "Hi there!"));

        assert_eq!(transcript.len(), 3);
        assert_eq!(transcript.session_id, Some("session-123".to_string()));
    }

    #[test]
    fn test_conversation_transcript_serialization() {
        let mut transcript = ConversationTranscript::with_session_id("session-456");
        transcript.push(ConversationTurn::new(TurnRole::User, "What's the weather?"));
        transcript.push(
            ConversationTurn::new(TurnRole::Agent, "Let me check for you.").with_tool_calls(vec![
                ToolCallEntry::new("get_weather", json!({"location": "New York"})),
            ]),
        );

        let json_str = serde_json::to_string_pretty(&transcript).unwrap();
        let deserialized: ConversationTranscript = serde_json::from_str(&json_str).unwrap();

        assert_eq!(transcript, deserialized);
    }

    #[test]
    fn test_turn_role_serialization() {
        let roles = vec![
            TurnRole::User,
            TurnRole::Agent,
            TurnRole::System,
            TurnRole::Tool,
        ];

        for role in roles {
            let json_str = serde_json::to_string(&role).unwrap();
            let deserialized: TurnRole = serde_json::from_str(&json_str).unwrap();
            assert_eq!(role, deserialized);
        }
    }

    #[test]
    fn test_optional_fields_serialization() {
        let entry = ToolCallEntry::new("test", json!({}));

        let json_str = serde_json::to_string(&entry).unwrap();
        let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Optional fields should not be present when None
        assert!(json_val.get("result").is_none());
        assert!(json_val.get("timestamp").is_none());
        assert!(json_val.get("duration_ms").is_none());
    }
}
