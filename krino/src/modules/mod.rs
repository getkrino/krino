//! Evaluation modules.
//!
//! This module contains all evaluation logic for different aspects of LLM outputs:
//! hallucination detection, groundedness checking, schema validation, etc.
//!
//! # Determinism
//!
//! All functions in this module are deterministic. Same input always produces
//! the same output.

pub mod groundedness;
pub mod hallucination;
pub mod policy_compliance;
pub mod schema;

// Modules will be added as they are implemented
// pub mod similarity;
// pub mod pii;
