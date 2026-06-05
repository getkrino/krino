//! Live Summarization Groundedness Checker
//!
//! This example demonstrates real-world groundedness verification for LLM summarization:
//! 1. Scrape text content from a live URL (using spider-rs)
//! 2. Generate a summary using Claude Sonnet (via Anthropic API)
//! 3. Verify the summary is grounded in the source text (using Krino NLI)
//!
//! # Use Case
//!
//! Many LLM applications summarize web content, documentation, articles, etc.
//! This tool ensures the LLM summary is faithful to the source material and
//! doesn't introduce hallucinated claims.
//!
//! # Setup
//!
//! ```bash
//! # 1. Download embedding model (for 180× speedup)
//! bash ../../scripts/download_embedding_model.sh
//!
//! # 2. Set your Anthropic API key
//! export ANTHROPIC_API_KEY="sk-ant-..."
//!
//! # 3. Ensure ONNX model is exported
//! cd ../../scripts && uv run export_deberta_onnx.py
//!
//! # 4. Run the example
//! cd examples/live_summarization
//! cargo run -- --url "https://en.wikipedia.org/wiki/Rust_(programming_language)"
//! ```
//!
//! # Example Output
//!
//! ```text
//! 🌐 Scraping: https://en.wikipedia.org/wiki/Rust_(programming_language)
//! ✓ Scraped 15,234 characters
//!
//! 🤖 Generating summary with Claude Sonnet...
//! ✓ Summary generated (512 tokens)
//!
//! 🔍 Verifying groundedness with Krino...
//! ✓ Verification complete (342ms)
//!
//! 📊 Groundedness Report:
//!   Faithfulness: 87.5%
//!   Supported claims: 7/8
//!   Contradicted claims: 0/8
//!   Neutral claims: 1/8
//!
//! ✅ Summary is grounded (≥70% threshold)
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use reqwest;
use serde_json::json;
use tracing::info;

/// CLI arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// URL to scrape and summarize
    #[arg(short, long)]
    url: String,

    /// Minimum faithfulness score to accept summary (0.0-1.0)
    #[arg(short, long, default_value = "0.7")]
    threshold: f64,

    /// Claude model to use
    #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Maximum tokens for summary
    #[arg(long, default_value = "1024")]
    max_tokens: u32,

    /// Krino API base URL (e.g. https://krino-alb-xxxxx.us-east-1.elb.amazonaws.com)
    #[arg(long, env = "KRINO_API_URL")]
    krino_api_url: String,

    /// Krino API key
    #[arg(long, env = "KRINO_API_KEY")]
    krino_api_key: String,
}


#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if it exists (silently ignore if not found)
    let _ = dotenvy::dotenv();

    // Initialize logging - silence ONNX Runtime's verbose BFC arena logs
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(
            tracing_subscriber::EnvFilter::new(
                "info,ort::logging=warn"  // Suppress ORT's internal allocator logs
            )
        )
        .init();

    let args = Args::parse();

    println!("🌐 Live Summarization Groundedness Checker");
    println!("===========================================\n");

    // =========================================================================
    // Step 1: Fetch website content (single page, no crawling)
    // =========================================================================
    println!("📥 Fetching: {}", args.url);

    let http_client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; KrinoGroundednessChecker/1.0)")
        .build()?;

    let response = http_client
        .get(&args.url)
        .send()
        .await
        .context("Failed to fetch URL")?;

    let html = response
        .text()
        .await
        .context("Failed to read response body")?;

    // Simple HTML stripping (production should use proper HTML parsing)
    let source_text = strip_html_tags(&html);

    println!("✓ Fetched {} characters\n", source_text.len());
    info!("Source preview: {}...", &source_text[..source_text.len().min(200)]);

    // =========================================================================
    // Step 2: Generate summary with Claude
    // =========================================================================
    println!("🤖 Generating summary with Claude {}...", args.model);

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY environment variable not set")?;

    let prompt = format!(
        "Please provide a concise, accurate summary of the following text. \
         Focus on the key facts and main points. Do not add information that \
         isn't present in the source text.\n\n{source_text}"
    );

    // Call Claude API directly using reqwest
    let client = reqwest::Client::new();
    let request_body = json!({
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    });

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request_body)
        .send()
        .await
        .context("Failed to call Claude API")?;

    let response_json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse Claude API response")?;

    // Log the full response for debugging
    info!("Claude API response: {}", serde_json::to_string_pretty(&response_json).unwrap_or_else(|_| "Failed to serialize".to_string()));

    // Check for API errors first
    if let Some(error) = response_json.get("error") {
        anyhow::bail!("Claude API error: {}", error);
    }

    // Extract summary from response
    let summary = response_json["content"][0]["text"]
        .as_str()
        .with_context(|| {
            format!(
                "Failed to extract text from Claude response. Response structure: {}",
                serde_json::to_string_pretty(&response_json).unwrap_or_else(|_| "Failed to serialize".to_string())
            )
        })?
        .to_string();

    if summary.is_empty() {
        anyhow::bail!("Claude returned empty summary");
    }

    println!("✓ Summary generated ({} chars)\n", summary.len());
    info!("Summary: {}", summary);

    // =========================================================================
    // Step 3: Verify groundedness via Krino API
    // =========================================================================
    println!("🔍 Verifying groundedness with Krino API...");

    let krino_client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true) // required for self-signed cert
        .build()?;

    let krino_response = krino_client
        .post(format!("{}/api/v1/evaluate", args.krino_api_url))
        .header("x-api-key", &args.krino_api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "context": [{"text": source_text}],
            "output": summary
        }))
        .send()
        .await
        .context("Failed to call Krino API")?;

    if !krino_response.status().is_success() {
        let status = krino_response.status();
        let body = krino_response.text().await.unwrap_or_default();
        anyhow::bail!("Krino API error {}: {}", status, body);
    }

    let verification_result: serde_json::Value = krino_response
        .json()
        .await
        .context("Failed to parse Krino API response")?;

    info!("Krino API response: {}", serde_json::to_string_pretty(&verification_result)?);

    let faithfulness_score = verification_result["score"]
        .as_f64()
        .context("Missing score in response")?;
    let claims = verification_result["claims"].as_array();
    let total_claims = claims.map(|c| c.len() as u64).unwrap_or(0);
    let supported_claims = claims.map(|c| c.iter().filter(|v| v["supported"].as_bool().unwrap_or(false)).count() as u64).unwrap_or(0);
    let contradicted_claims = claims.map(|c| c.iter().filter(|v| v["verdict"].as_str() == Some("contradiction")).count() as u64).unwrap_or(0);
    let neutral_claims = total_claims.saturating_sub(supported_claims).saturating_sub(contradicted_claims);
    let latency_ms = verification_result["meta"]["latency_ms"].as_f64().unwrap_or(0.0);

    println!("✓ Verification complete ({:.0}ms)\n", latency_ms);

    // =========================================================================
    // Step 4: Generate report
    // =========================================================================
    println!("📊 Groundedness Report:");
    println!("   Faithfulness: {:.1}%", faithfulness_score * 100.0);
    println!("   ✓ Supported claims: {}/{}", supported_claims, total_claims);
    println!("   ✗ Contradicted claims: {}/{}", contradicted_claims, total_claims);
    println!("   ∅ Neutral claims: {}/{}", neutral_claims, total_claims);
    println!("   Latency: {:.0}ms", latency_ms);

    // Final verdict
    println!();
    if faithfulness_score >= args.threshold {
        println!("✅ Summary is grounded (≥{:.0}% threshold)", args.threshold * 100.0);
        println!("\n📄 Verified Summary:");
        println!("{}", summary);
    } else {
        println!("❌ Summary contains hallucinations (<{:.0}% threshold)", args.threshold * 100.0);
        println!("   Consider regenerating with stricter instructions.");
    }

    Ok(())
}

/// Simple HTML tag stripper (production should use proper HTML parser)
fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut inside_tag = false;
    let mut inside_script_or_style = false;
    let mut tag_content = String::new();

    for ch in html.chars() {
        if ch == '<' {
            inside_tag = true;
            tag_content.clear();
        } else if ch == '>' {
            inside_tag = false;

            // Check if we just exited a script or style tag
            let tag_lower = tag_content.to_lowercase();
            if tag_lower.starts_with("script") || tag_lower.starts_with("style") {
                inside_script_or_style = true;
            } else if tag_lower.starts_with("/script") || tag_lower.starts_with("/style") {
                inside_script_or_style = false;
            }

            tag_content.clear();
        } else if inside_tag {
            tag_content.push(ch);
        } else if !inside_script_or_style {
            result.push(ch);
        }
    }

    // Clean up whitespace
    result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        let html = "<p>Hello <b>world</b>!</p>";
        assert_eq!(strip_html_tags(html), "Hello world !");

        let html_with_script = "<p>Keep this</p><script>alert('remove');</script><p>And this</p>";
        assert_eq!(strip_html_tags(html_with_script), "Keep this And this");
    }
}
