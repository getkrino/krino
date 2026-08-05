//! Krino CLI binary.
//!
//! This is the command-line interface for Krino evaluation engine.

use krino::{KrinoConfig, KrinoEngine, init_tracing};
use std::path::Path;

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};

#[cfg(feature = "cli")]
#[derive(Parser)]
#[command(name = "krino")]
#[command(about = "Deterministic LLM evaluation engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "cli")]
#[derive(Subcommand)]
enum Commands {
    /// Display version information
    Version,

    /// Validate configuration file
    ValidateConfig {
        /// Path to configuration file
        #[arg(short, long, default_value = "krino.json")]
        config: String,
    },

    /// Show default configuration
    ShowConfig,

    /// Evaluate text for hallucinations
    EvalHallucination {
        /// Text to evaluate for hallucinations (legacy mode: treats entire text as answer)
        #[arg(short, long)]
        text: Option<String>,

        /// Path to text file to evaluate
        #[arg(short, long)]
        file: Option<String>,

        /// Context/source document for RAG hallucination detection
        #[arg(short, long)]
        context: Option<String>,

        /// Question asked (optional, for RAG mode)
        #[arg(short, long)]
        question: Option<String>,

        /// Answer to verify against context (for RAG mode)
        #[arg(short, long)]
        answer: Option<String>,

        /// Path to model directory (containing model files and tokenizer)
        #[arg(short = 'p', long)]
        model_path: String,

        /// Confidence threshold for flagging hallucinations (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        threshold: f64,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        output: String,
    },

    /// Validate JSON against a schema
    ValidateSchema {
        /// JSON string to validate
        #[arg(short, long)]
        json: Option<String>,

        /// Path to JSON file to validate
        #[arg(short, long)]
        file: Option<String>,

        /// Path to JSON Schema file
        #[arg(short, long)]
        schema: String,

        /// Maximum nesting depth (0 = unlimited)
        #[arg(long, default_value = "10")]
        max_depth: usize,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        output: String,
    },
}

#[cfg(feature = "cli")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let cli = Cli::parse();

    match cli.command {
        Commands::Version => {
            println!("Krino v{}", KrinoEngine::version());
            println!("Deterministic LLM evaluation engine");
            println!();
            println!("Same inputs. Same results. Every time.");
        }

        Commands::ValidateConfig { config } => {
            println!("Validating configuration: {}", config);
            let cfg = KrinoConfig::from_file(&config)?;
            cfg.validate()?;
            println!("✓ Configuration is valid");
            println!("  - {} models configured", cfg.models.len());
            println!("  - Max latency: {}ms", cfg.performance.max_latency_ms);
        }

        Commands::ShowConfig => {
            let config = KrinoConfig::default();
            let json = serde_json::to_string_pretty(&config)?;
            println!("{}", json);
        }

        Commands::EvalHallucination {
            text,
            file,
            context,
            question,
            answer,
            model_path,
            threshold,
            output,
        } => {
            handle_eval_hallucination(EvalHallucinationParams {
                text,
                file,
                context,
                question,
                answer,
                model_path,
                threshold,
                output_format: output,
            })?;
        }

        Commands::ValidateSchema {
            json,
            file,
            schema,
            max_depth,
            output,
        } => {
            handle_validate_schema(json, file, schema, max_depth, output)?;
        }
    }

    Ok(())
}

#[cfg(feature = "cli")]
struct EvalHallucinationParams {
    text: Option<String>,
    file: Option<String>,
    context: Option<String>,
    question: Option<String>,
    answer: Option<String>,
    model_path: String,
    threshold: f64,
    output_format: String,
}

#[cfg(feature = "cli")]
fn handle_eval_hallucination(
    params: EvalHallucinationParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let EvalHallucinationParams {
        text,
        file,
        context,
        question,
        answer,
        model_path,
        threshold,
        output_format,
    } = params;
    use krino::models::backends::CandleBackend;
    use krino::modules::hallucination::{HallucinationConfig, HallucinationDetector};
    use std::sync::Arc;
    use tokenizers::Tokenizer;

    // Determine mode: RAG (context/question/answer) or legacy (text/file)
    let is_rag_mode = context.is_some() || answer.is_some();

    if is_rag_mode && (text.is_some() || file.is_some()) {
        return Err(
            "Cannot mix RAG mode (--context/--answer) with legacy mode (--text/--file)".into(),
        );
    }

    // Get input text (legacy mode)
    let input_text = if !is_rag_mode {
        match (text, file) {
            (Some(t), None) => Some(t),
            (None, Some(f)) => Some(std::fs::read_to_string(f)?),
            (Some(_), Some(_)) => {
                return Err("Cannot specify both --text and --file".into());
            }
            (None, None) => {
                return Err("Must specify either --text/--file or --context/--answer".into());
            }
        }
    } else {
        None
    };

    println!("🔍 Loading model from: {model_path}");

    // Load tokenizer
    let tokenizer_path = Path::new(&model_path).join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "Tokenizer not found at: {}. Expected tokenizer.json in model directory.",
            tokenizer_path.display()
        )
        .into());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    // Load model with Candle backend
    println!("📦 Using Candle ModernBERT backend");
    let model_path_buf = Path::new(&model_path);
    let backend: Arc<dyn krino::models::TokenClassifier> =
        Arc::new(CandleBackend::from_pretrained(model_path_buf, 2)?);

    // Create detector with config
    let config = HallucinationConfig {
        threshold,
        ..Default::default()
    };

    let detector = HallucinationDetector::new(backend, tokenizer, config);

    println!("✅ Model loaded successfully");

    // Run detection based on mode
    let result = if is_rag_mode {
        let ctx = context.ok_or("RAG mode requires --context")?;
        let ans = answer.ok_or("RAG mode requires --answer")?;
        let q = question.unwrap_or_default();

        println!(
            "🧪 Analyzing RAG output (context: {} chars, answer: {} chars)...\n",
            ctx.len(),
            ans.len()
        );

        detector.detect_rag(&ctx, &q, &ans)?
    } else {
        let text = input_text.unwrap();
        println!("🧪 Analyzing text ({} chars)...\n", text.len());
        detector.detect(&text)?
    };

    // Output results
    match output_format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{json}");
        }
        _ => {
            println!("📊 Results:");
            println!("  Total tokens: {}", result.total_tokens);
            println!("  Hallucinated tokens: {}", result.hallucinated_tokens);
            println!("  Aggregate score: {:.2}%", result.aggregate_score * 100.0);
            println!("  Latency: {:.2}ms", result.latency_ms);
            println!();

            if result.hallucinated_spans.is_empty() {
                println!("✅ No hallucinations detected!");
            } else {
                println!(
                    "⚠️  Detected {} hallucination span(s):",
                    result.hallucinated_spans.len()
                );
                println!();

                for (idx, span) in result.hallucinated_spans.iter().enumerate() {
                    println!("  {}. \"{}\"", idx + 1, span.text);
                    println!("     Position: {}..{}", span.start, span.end);
                    println!("     Confidence: {:.2}%", span.confidence * 100.0);
                    println!("     Reason: {}", span.evidence_gap);
                    println!();
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cli")]
fn handle_validate_schema(
    json_str: Option<String>,
    file_path: Option<String>,
    schema_path: String,
    max_depth: usize,
    output_format: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use krino::modules::schema::{SchemaConfig, SchemaValidator};

    // Get input JSON
    let json_input = match (json_str, file_path) {
        (Some(j), None) => j,
        (None, Some(f)) => std::fs::read_to_string(f)?,
        (Some(_), Some(_)) => {
            return Err("Cannot specify both --json and --file".into());
        }
        (None, None) => {
            return Err("Must specify either --json or --file".into());
        }
    };

    println!("📋 Loading schema from: {schema_path}");

    // Load schema
    let schema_json = std::fs::read_to_string(&schema_path)?;
    let schema: serde_json::Value = serde_json::from_str(&schema_json)?;

    // Create validator
    let config = SchemaConfig {
        schema,
        strict_mode: false,
        max_depth: if max_depth == 0 {
            None
        } else {
            Some(max_depth)
        },
    };

    let validator = SchemaValidator::new(config)?;
    println!("✅ Schema loaded successfully");
    println!();

    // Validate
    println!("🧪 Validating JSON ({} chars)...\n", json_input.len());
    let result = validator.validate(&json_input)?;

    // Output results
    match output_format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{json}");
        }
        _ => {
            println!("📊 Results:");
            println!("  Valid: {}", if result.valid { "✅ Yes" } else { "❌ No" });
            println!("  JSON Parse Success: {}", result.json_parse_success);
            println!("  Nesting Depth: {}", result.nesting_depth);
            println!("  Error Count: {}", result.error_count);
            println!("  Latency: {:.2}ms", result.latency_ms);
            println!();

            if result.valid {
                println!("✅ JSON is valid according to schema!");
            } else {
                if !result.json_parse_success {
                    println!("❌ JSON Parsing Failed:");
                    if let Some(err) = &result.json_parse_error {
                        println!("   {err}");
                    }
                } else {
                    println!("❌ Found {} validation error(s):", result.errors.len());
                    println!();

                    for (idx, error) in result.errors.iter().enumerate() {
                        println!("  {}. Path: {}", idx + 1, error.path);
                        println!("     Error: {}", error.message);
                        println!("     Kind: {}", error.kind);
                        println!();
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("Krino CLI requires the 'cli' feature flag.");
    eprintln!("Install with: cargo install krino --features cli");
    std::process::exit(1);
}
