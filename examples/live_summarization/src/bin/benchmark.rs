use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use krino::models::backends::candle::CandleEmbeddingBackend;
use krino::models::backends::onnx::OnnxSequenceClassifier;
use krino::models::inference::EmbeddingSimilarity;
use krino::modules::groundedness::{GroundednessChecker, GroundednessConfig};

const ITERATIONS: usize = 10;

const PRESET_URLS: &[(&str, &str)] = &[
    ("Rust",          "https://en.wikipedia.org/wiki/Rust_(programming_language)"),
    ("Lake Michigan", "https://en.wikipedia.org/wiki/Lake_Michigan"),
];

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum Mode {
    Api,
    Local,
    Both,
}

#[derive(Parser, Debug)]
#[command(about = "Benchmark local Krino vs deployed API on the full live summarization pipeline")]
struct Args {
    /// Which target(s) to benchmark
    #[arg(long, default_value = "api")]
    mode: Mode,

    /// Claude model to use for summarization
    #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Maximum tokens for Claude summary
    #[arg(long, default_value = "1024")]
    max_tokens: u32,

    /// Krino API base URL
    #[arg(long, env = "KRINO_API_URL")]
    krino_api_url: String,

    /// Krino API key
    #[arg(long, env = "KRINO_API_KEY")]
    krino_api_key: String,

    /// Path to ONNX NLI model (only needed for --mode local or both)
    #[arg(long, default_value = "../../models/nli-small-onnx")]
    nli_model_path: String,

    /// Path to embedding model (only needed for --mode local or both)
    #[arg(long, default_value = "../../models/all-MiniLM-L6-v2")]
    embedding_model_path: String,
}

fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)];
    let p99 = sorted[((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1)];
    (mean, p95, p99)
}

fn print_stats(label: &str, samples: &[f64]) {
    let (mean, p95, p99) = stats(samples);
    println!("  {label}");
    println!("    mean  {mean:>8.1}ms");
    println!("    p95   {p95:>8.1}ms");
    println!("    p99   {p99:>8.1}ms");
    println!("    min   {:>8.1}ms", samples.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("    max   {:>8.1}ms", samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
}

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

    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct TopicInput {
    label: String,
    source_text: String,
    summary: String,
    summarization_ms: f64,
}

/// Fetches an article's body text, stripped of nav chrome and HTML tags.
///
/// For Wikipedia URLs (`*.wikipedia.org/wiki/<TITLE>`) this calls the MediaWiki
/// `parse` API to get *only* the article HTML — without navigation, footer,
/// or sidebar markup. Stripping then yields clean text. Raw-scraping the
/// full Wikipedia page produced massive unsplittable "sentences" of mashed
/// nav text (~2800 chars each), which doubled the engine's embedding-pass
/// time on the same article (~760ms → ~1730ms, measured 2026-05-19).
///
/// Non-Wikipedia URLs fall back to a raw GET + tag strip so a developer can
/// still benchmark against arbitrary pages.
async fn fetch_article_text(http_client: &reqwest::Client, url: &str) -> Result<String> {
    if let Some(title) = wikipedia_title_from_url(url) {
        let api_url = format!(
            "https://en.wikipedia.org/w/api.php?action=parse&page={title}&prop=text&formatversion=2&format=json"
        );
        let resp: serde_json::Value = http_client
            .get(&api_url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch Wikipedia parse API for {title}"))?
            .json()
            .await
            .context("Failed to parse Wikipedia parse API response")?;
        if let Some(err) = resp.get("error") {
            anyhow::bail!("Wikipedia parse API error: {err}");
        }
        let article_html = resp["parse"]["text"]
            .as_str()
            .context("Wikipedia parse API response missing .parse.text")?;
        return Ok(strip_html_tags(article_html));
    }

    // Non-Wikipedia fallback: raw page scrape.
    let html = http_client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to fetch {url}"))?
        .text()
        .await
        .context("Failed to read response body")?;
    Ok(strip_html_tags(&html))
}

/// Extracts the article title from a Wikipedia URL, or `None` for non-Wikipedia URLs.
///
/// Examples:
/// - `https://en.wikipedia.org/wiki/Rust_(programming_language)` → `Some("Rust_(programming_language)")`
/// - `https://example.com/foo` → `None`
fn wikipedia_title_from_url(url: &str) -> Option<String> {
    // Find scheme separator.
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    // Split host from path.
    let (host, path) = after_scheme.split_once('/')?;
    if !host.ends_with("wikipedia.org") {
        return None;
    }
    // Find `/wiki/<title>` — title runs to next slash or query/fragment.
    let after_wiki = path.strip_prefix("wiki/")?;
    let title_end = after_wiki
        .find(['/', '?', '#'])
        .unwrap_or(after_wiki.len());
    let title = &after_wiki[..title_end];
    if title.is_empty() {
        return None;
    }
    Some(title.to_string())
}

async fn fetch_and_summarize(
    label: &str,
    url: &str,
    model: &str,
    max_tokens: u32,
    http_client: &reqwest::Client,
    claude_client: &reqwest::Client,
    api_key: &str,
) -> Result<TopicInput> {
    eprint!("  [{label}] fetching... ");

    let source_text = fetch_article_text(http_client, url).await?;

    eprint!("summarizing... ");
    let prompt = format!(
        "Please provide a concise, accurate summary of the following text. \
         Focus on the key facts and main points. Do not add information that \
         isn't present in the source text.\n\n{source_text}"
    );

    let t_claude = Instant::now();
    let resp_json: serde_json::Value = claude_client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }))
        .send()
        .await
        .context("Failed to call Claude API")?
        .json()
        .await
        .context("Failed to parse Claude response")?;
    let summarization_ms = t_claude.elapsed().as_secs_f64() * 1000.0;

    if let Some(error) = resp_json.get("error") {
        anyhow::bail!("Claude API error for [{label}]: {error}");
    }

    let summary = resp_json["content"][0]["text"]
        .as_str()
        .context("Failed to extract summary from Claude response")?
        .to_string();

    // Cap context sent to the API — large articles (80k+ chars) cause OOM on the EC2 instance
    const MAX_CONTEXT_CHARS: usize = 40_000;
    let source_text = if source_text.len() > MAX_CONTEXT_CHARS {
        source_text[..MAX_CONTEXT_CHARS].to_string()
    } else {
        source_text
    };

    eprintln!("done ({} chars context, {} chars summary, {summarization_ms:.0}ms claude)",
        source_text.len(), summary.len());

    Ok(TopicInput { label: label.to_string(), source_text, summary, summarization_ms })
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();

    let run_local = matches!(args.mode, Mode::Local | Mode::Both);
    let run_api = matches!(args.mode, Mode::Api | Mode::Both);

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY environment variable not set")?;

    let http_client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; KrinoGroundednessChecker/1.0)")
        .danger_accept_invalid_certs(true)
        .build()?;
    let claude_client = reqwest::Client::new();

    // Phase 1: fetch + summarize (progress to stderr, silent on stdout)
    eprintln!("Preparing {} topics...", PRESET_URLS.len());
    let mut inputs: Vec<TopicInput> = Vec::new();
    for (i, (label, url)) in PRESET_URLS.iter().enumerate() {
        if i > 0 {
            eprintln!("  [waiting 10s between Claude calls...]");
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        }
        let input = fetch_and_summarize(
            label, url, &args.model, args.max_tokens,
            &http_client, &claude_client, &api_key,
        ).await?;
        inputs.push(input);
    }

    // Phase 2: load local models if needed
    let checker = if run_local {
        eprint!("Loading local models... ");
        let nli_path = Path::new(&args.nli_model_path);
        if !nli_path.join("model.onnx").exists() {
            anyhow::bail!(
                "ONNX model not found at {}. Run: cd scripts && uv run export_deberta_onnx.py",
                nli_path.display()
            );
        }
        let nli_backend = Arc::new(OnnxSequenceClassifier::from_pretrained(nli_path)?);

        let embedding_path = Path::new(&args.embedding_model_path);
        if !embedding_path.join("model.safetensors").exists() {
            anyhow::bail!(
                "Embedding model not found at {}. Run: bash scripts/download_embedding_model.sh",
                embedding_path.display()
            );
        }
        let embedding_backend = Arc::new(CandleEmbeddingBackend::from_pretrained(embedding_path)?)
            as Arc<dyn EmbeddingSimilarity>;

        let config = GroundednessConfig {
            contradiction_threshold: 0.7,
            treat_neutral_as_unsupported: false,
            top_k_context: 5,
            include_entailment_matrix: false,
            flag_compound_claims: true,
            ..Default::default()
        };
        eprintln!("done");
        Some(GroundednessChecker::new(nli_backend, embedding_backend, config))
    } else {
        None
    };

    // Phase 3: benchmark
    eprintln!("Running {ITERATIONS} iterations per topic...\n");

    struct TopicResult {
        label: String,
        summarization_ms: f64,
        local_samples: Vec<f64>,
        api_wall_samples: Vec<f64>,     // wall-clock per call (includes network)
        api_server_samples: Vec<f64>,   // server-side latency_ms from meta
        api_split_samples: Vec<f64>,    // server-side meta.split_ms (sentence splitting)
        api_embed_samples: Vec<f64>,    // server-side meta.embedding_ms (pre-filter)
        api_nli_samples: Vec<f64>,      // server-side meta.nli_ms (NLI batch inference)
        local_score: f64,
        api_score: f64,
        api_nli_calls: usize,
        api_model: String,
        api_engine_version: String,
    }

    let mut results: Vec<TopicResult> = Vec::new();

    for input in &inputs {
        let mut local_samples = Vec::new();
        let mut local_score = 0.0_f64;
        if let Some(ref c) = checker {
            local_samples = Vec::with_capacity(ITERATIONS);
            for _ in 0..ITERATIONS {
                let t = Instant::now();
                let result = c.check(&input.source_text, &input.summary)?;
                local_samples.push(t.elapsed().as_secs_f64() * 1000.0);
                local_score = result.faithfulness_score;
            }
        }

        let mut api_wall_samples = Vec::new();
        let mut api_server_samples = Vec::new();
        let mut api_split_samples = Vec::new();
        let mut api_embed_samples = Vec::new();
        let mut api_nli_samples = Vec::new();
        let mut api_score = 0.0_f64;
        let mut api_nli_calls = 0usize;
        let mut api_model = String::new();
        let mut api_engine_version = String::new();

        if run_api {
            api_wall_samples = Vec::with_capacity(ITERATIONS);
            api_server_samples = Vec::with_capacity(ITERATIONS);
            api_split_samples = Vec::with_capacity(ITERATIONS);
            api_embed_samples = Vec::with_capacity(ITERATIONS);
            api_nli_samples = Vec::with_capacity(ITERATIONS);
            for _ in 0..ITERATIONS {
                let t = Instant::now();
                let resp = http_client
                    .post(format!("{}/api/v1/evaluate", args.krino_api_url))
                    .header("x-api-key", &args.krino_api_key)
                    .header("Content-Type", "application/json")
                    .json(&json!({
                        "context": [{"text": input.source_text}],
                        "output": input.summary
                    }))
                    .send()
                    .await
                    .context("Failed to call Krino API")?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    anyhow::bail!("Krino API error {status}: {body}");
                }

                let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
                let body: serde_json::Value = resp.json().await.context("Failed to parse API response")?;
                api_score = body["score"].as_f64().unwrap_or(0.0);
                api_nli_calls = body["meta"]["inference_calls"].as_u64().unwrap_or(0) as usize;
                api_model = body["meta"]["model"].as_str().unwrap_or("unknown").to_string();
                api_engine_version = body["meta"]["engine_version"].as_str().unwrap_or("unknown").to_string();
                let server_ms = body["meta"]["latency_ms"].as_f64().unwrap_or(0.0);
                let split_ms = body["meta"]["split_ms"].as_f64().unwrap_or(0.0);
                let embed_ms = body["meta"]["embedding_ms"].as_f64().unwrap_or(0.0);
                let nli_ms = body["meta"]["nli_ms"].as_f64().unwrap_or(0.0);
                api_wall_samples.push(wall_ms);
                api_server_samples.push(server_ms);
                api_split_samples.push(split_ms);
                api_embed_samples.push(embed_ms);
                api_nli_samples.push(nli_ms);
            }
        }

        results.push(TopicResult {
            label: input.label.clone(),
            summarization_ms: input.summarization_ms,
            local_samples,
            api_wall_samples,
            api_server_samples,
            api_split_samples,
            api_embed_samples,
            api_nli_samples,
            local_score,
            api_score,
            api_nli_calls,
            api_model,
            api_engine_version,
        });
    }

    // ── Terminal output ────────────────────────────────────────────────────────
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    println!("\nKrino Benchmark Results");
    println!("mode={:?}  topics={}  iterations={ITERATIONS}  {timestamp}\n", args.mode, PRESET_URLS.len());

    let mut all_local: Vec<f64> = Vec::new();
    let mut all_api_wall: Vec<f64> = Vec::new();
    let mut all_api_server: Vec<f64> = Vec::new();

    for r in &results {
        println!("── {} ──", r.label);
        println!("  summarization (Claude)  {:.0}ms", r.summarization_ms);
        println!();

        if !r.local_samples.is_empty() {
            print_stats("Local (in-process)", &r.local_samples);
            println!("    score {:.4}", r.local_score);
            println!();
        }

        if !r.api_wall_samples.is_empty() {
            // Cold vs warm split
            let cold = r.api_wall_samples[0];
            let warm = if r.api_wall_samples.len() > 1 { &r.api_wall_samples[1..] } else { &r.api_wall_samples[..] };
            let (warm_mean, warm_p95, warm_p99) = stats(warm);

            print_stats("API   (wall-clock + network)", &r.api_wall_samples);
            println!("    cold (iter 1)  {cold:.0}ms");
            println!("    warm mean      {warm_mean:.0}ms  p95={warm_p95:.0}ms  p99={warm_p99:.0}ms");

            print_stats("API   (server-side only)", &r.api_server_samples);
            let network_overhead = r.api_wall_samples.iter().zip(r.api_server_samples.iter())
                .map(|(w, s)| w - s)
                .collect::<Vec<_>>();
            let (net_mean, _, _) = stats(&network_overhead);
            println!("    network overhead  {net_mean:.0}ms avg");
            // Server-side breakdown: split (sentence segmentation) + embed
            // (claim/context vectorization for pre-filter) + nli (per-pair
            // entailment inference). These sum to roughly the server-side
            // total; remainder is post-processing, allocation, and serde.
            let (split_mean, _, _) = stats(&r.api_split_samples);
            let (embed_mean, _, _) = stats(&r.api_embed_samples);
            let (nli_mean, _, _) = stats(&r.api_nli_samples);
            println!(
                "    server breakdown  split={split_mean:.0}ms  embed={embed_mean:.0}ms  nli={nli_mean:.0}ms"
            );
            println!("    score {:.4}  nli_calls={}  model={}  engine={}",
                r.api_score, r.api_nli_calls, r.api_model, r.api_engine_version);
        }

        if !r.local_samples.is_empty() && !r.api_wall_samples.is_empty() {
            let diff = (r.local_score - r.api_score).abs();
            let agreement = if diff < 0.01 { "✅" } else { "⚠️ " };
            println!("    score agreement {agreement} Δ{diff:.4}");
        }
        println!();
        all_local.extend(&r.local_samples);
        all_api_wall.extend(&r.api_wall_samples);
        all_api_server.extend(&r.api_server_samples);
    }

    if PRESET_URLS.len() > 1 {
        println!("── Aggregate ({} topics × {ITERATIONS}) ──", PRESET_URLS.len());
        if !all_local.is_empty() {
            print_stats("Local", &all_local);
            println!();
        }
        if !all_api_wall.is_empty() {
            print_stats("API wall-clock", &all_api_wall);
            println!();
            print_stats("API server-side", &all_api_server);
        }
        if !all_local.is_empty() && !all_api_wall.is_empty() {
            let (local_mean, _, _) = stats(&all_local);
            let (api_mean, _, _) = stats(&all_api_wall);
            let overhead_pct = ((api_mean - local_mean) / local_mean) * 100.0;
            println!("    network overhead +{overhead_pct:.1}% ({:.1}ms avg)", api_mean - local_mean);
        }
    }

    // ── Markdown report ────────────────────────────────────────────────────────
    let report_path = "benchmark_results.md";
    let mut md = String::new();

    md.push_str(&format!("## Run: {timestamp}\n\n"));
    md.push_str(&format!("- **Mode**: {:?}\n", args.mode));
    md.push_str(&format!("- **Topics**: {}\n", PRESET_URLS.len()));
    md.push_str(&format!("- **Iterations**: {ITERATIONS} per topic\n"));
    if !results.is_empty() && !results[0].api_engine_version.is_empty() {
        md.push_str(&format!("- **Engine**: {}\n", results[0].api_engine_version));
        md.push_str(&format!("- **Model**: {}\n", results[0].api_model));
    }
    md.push('\n');

    for r in &results {
        md.push_str(&format!("### {}\n\n", r.label));
        md.push_str(&format!("- **Summarization (Claude)**: {:.0}ms\n\n", r.summarization_ms));

        if !r.api_wall_samples.is_empty() {
            let cold = r.api_wall_samples[0];
            let warm = if r.api_wall_samples.len() > 1 { &r.api_wall_samples[1..] } else { &r.api_wall_samples[..] };
            let (warm_mean, warm_p95, warm_p99) = stats(warm);
            let (wall_mean, wall_p95, wall_p99) = stats(&r.api_wall_samples);
            let wall_min = r.api_wall_samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let wall_max = r.api_wall_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let (srv_mean, srv_p95, srv_p99) = stats(&r.api_server_samples);
            let srv_min = r.api_server_samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let srv_max = r.api_server_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let network_overhead: Vec<f64> = r.api_wall_samples.iter().zip(r.api_server_samples.iter())
                .map(|(w, s)| w - s).collect();
            let (net_mean, _, _) = stats(&network_overhead);

            md.push_str("| Metric | mean | p95 | p99 | min | max |\n");
            md.push_str("|--------|-----:|----:|----:|----:|----:|\n");
            md.push_str(&format!("| Wall-clock | {wall_mean:.0}ms | {wall_p95:.0}ms | {wall_p99:.0}ms | {wall_min:.0}ms | {wall_max:.0}ms |\n"));
            md.push_str(&format!("| Server-side | {srv_mean:.0}ms | {srv_p95:.0}ms | {srv_p99:.0}ms | {srv_min:.0}ms | {srv_max:.0}ms |\n"));
            md.push('\n');
            md.push_str(&format!("- **Cold (iter 1)**: {cold:.0}ms  \n"));
            md.push_str(&format!("- **Warm mean**: {warm_mean:.0}ms  p95={warm_p95:.0}ms  p99={warm_p99:.0}ms  \n"));
            md.push_str(&format!("- **Network overhead**: {net_mean:.0}ms avg  \n"));
            md.push_str(&format!("- **Score**: {:.4}  **NLI calls**: {}  \n\n", r.api_score, r.api_nli_calls));
        }

        if !r.local_samples.is_empty() {
            let (mean, p95, p99) = stats(&r.local_samples);
            let min = r.local_samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = r.local_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            md.push_str("| Metric | mean | p95 | p99 | min | max |\n");
            md.push_str("|--------|-----:|----:|----:|----:|----:|\n");
            md.push_str(&format!("| Local | {mean:.0}ms | {p95:.0}ms | {p99:.0}ms | {min:.0}ms | {max:.0}ms |\n\n"));
            md.push_str(&format!("- **Score**: {:.4}  \n\n", r.local_score));
        }
    }

    if !all_api_wall.is_empty() || !all_local.is_empty() {
        md.push_str(&format!("### Aggregate ({} topics × {ITERATIONS})\n\n", PRESET_URLS.len()));
        md.push_str("| Target | mean | p95 | p99 | min | max |\n");
        md.push_str("|--------|-----:|----:|----:|----:|----:|\n");
        if !all_api_wall.is_empty() {
            let (mean, p95, p99) = stats(&all_api_wall);
            let min = all_api_wall.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = all_api_wall.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            md.push_str(&format!("| API wall-clock | {mean:.0}ms | {p95:.0}ms | {p99:.0}ms | {min:.0}ms | {max:.0}ms |\n"));
            let (mean, p95, p99) = stats(&all_api_server);
            let min = all_api_server.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = all_api_server.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            md.push_str(&format!("| API server-side | {mean:.0}ms | {p95:.0}ms | {p99:.0}ms | {min:.0}ms | {max:.0}ms |\n"));
        }
        if !all_local.is_empty() {
            let (mean, p95, p99) = stats(&all_local);
            let min = all_local.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = all_local.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            md.push_str(&format!("| Local | {mean:.0}ms | {p95:.0}ms | {p99:.0}ms | {min:.0}ms | {max:.0}ms |\n"));
        }
        md.push('\n');
    }

    md.push_str("---\n\n");

    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(report_path)
        .context("Failed to open benchmark_results.md")?;
    file.write_all(md.as_bytes())
        .context("Failed to write benchmark_results.md")?;

    eprintln!("Results appended to {report_path}");

    Ok(())
}
