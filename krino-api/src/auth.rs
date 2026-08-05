use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use subtle::ConstantTimeEq;

use crate::config::ApiConfig;
use crate::state::AppState;

pub struct ApiKeyStore {
    /// key_hash → customer_id
    keys: BTreeMap<String, String>,
}

impl ApiKeyStore {
    pub fn from_config(config: &ApiConfig) -> anyhow::Result<Self> {
        let mut keys = BTreeMap::new();
        for key_entry in &config.auth.api_keys {
            let hash = hash_key(&key_entry.key);
            keys.insert(hash, key_entry.customer_id.clone());
        }
        tracing::info!("Loaded {} API keys", keys.len());
        Ok(Self { keys })
    }

    /// Validates an API key and returns the customer ID.
    /// Uses constant-time comparison to prevent timing attacks.
    pub fn validate(&self, provided_key: &str) -> Option<&str> {
        let provided_hash = hash_key(provided_key);
        // Iterate all keys with constant-time comparison to prevent
        // timing side-channels that reveal which keys exist.
        for (stored_hash, customer_id) in &self.keys {
            if stored_hash
                .as_bytes()
                .ct_eq(provided_hash.as_bytes())
                .into()
            {
                return Some(customer_id.as_str());
            }
        }
        None
    }
}

fn hash_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

/// Middleware that extracts and validates the API key from the
/// Authorization header or x-api-key header.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let key = req
        .headers()
        .get("x-api-key")
        .or_else(|| req.headers().get("authorization"))
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim_start_matches("Bearer ").trim());

    let Some(key) = key else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    let Some(customer_id) = state.auth.validate(key) else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    // Insert customer ID into request extensions for downstream use
    req.extensions_mut()
        .insert(CustomerId(customer_id.to_string()));

    Ok(next.run(req).await)
}

#[derive(Clone)]
pub struct CustomerId(pub String);
