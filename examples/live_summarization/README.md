# Live Summarization Groundedness Checker

Real-world demonstration of groundedness verification for LLM-generated summaries.

## What This Does

1. **Scrapes** live web content from any URL (using `spider-rs`)
2. **Summarizes** the content using Claude Sonnet (via Anthropic API)
3. **Verifies** the summary is grounded in the source text (using Krino NLI)

## Use Case

Many LLM applications summarize web content (articles, documentation, research papers, etc.). This tool ensures the LLM summary is **faithful to the source material** and doesn't introduce hallucinated claims.

## Setup

### 1. Set API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### 2. Ensure ONNX Model is Exported

```bash
cd ../../scripts
uv run export_deberta_onnx.py
```

This creates `models/deberta-nli-onnx/` with the quantized NLI model.

### 3. Run the Example

```bash
cd examples/live_summarization
cargo run -- --url "https://en.wikipedia.org/wiki/Rust_(programming_language)"
```

## Example Output

```
🌐 Live Summarization Groundedness Checker
===========================================

📥 Scraping: https://en.wikipedia.org/wiki/Rust_(programming_language)
✓ Scraped 15,234 characters

🤖 Generating summary with Claude claude-sonnet-4-20250514...
✓ Summary generated (512 chars)

🔍 Verifying groundedness with Krino...
✓ Verification complete (342ms)

📊 Groundedness Report:
   Faithfulness: 87.5%
   ✓ Supported claims: 7/8
   ✗ Contradicted claims: 0/8
   ∅ Neutral claims: 1/8
   Latency: 342ms
   NLI calls: 24

✅ Summary is grounded (≥70% threshold)

📄 Verified Summary:
Rust is a multi-paradigm programming language emphasizing performance,
type safety, and concurrency. It was designed by Graydon Hoare at
Mozilla Research...
```

## CLI Options

```bash
cargo run -- --help

Options:
  -u, --url <URL>                    URL to scrape and summarize
  -t, --threshold <THRESHOLD>        Minimum faithfulness (0.0-1.0) [default: 0.7]
  -m, --model <MODEL>                Claude model [default: claude-sonnet-4-20250514]
      --max-tokens <MAX_TOKENS>      Max tokens for summary [default: 1024]
      --nli-model-path <PATH>        Path to ONNX NLI model [default: ../../models/deberta-nli-onnx]
```

## Example URLs to Try

### Wikipedia Articles
```bash
cargo run -- --url "https://en.wikipedia.org/wiki/Rust_(programming_language)"
cargo run -- --url "https://en.wikipedia.org/wiki/Large_language_model"
```

### Documentation Sites
```bash
cargo run -- --url "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html"
```

### News Articles
```bash
cargo run -- --url "https://www.bbc.com/news/technology"
```

## Understanding the Output

### Faithfulness Score
- **≥90%**: Excellent - summary is highly grounded
- **70-89%**: Good - summary is mostly grounded, minor neutral claims
- **50-69%**: Fair - some unsupported claims, review recommended
- **<50%**: Poor - significant hallucinations detected

### Claim Labels
- **Entailment (✓)**: Claim is supported by the source text
- **Neutral (∅)**: Claim is plausible but not explicitly stated
- **Contradiction (✗)**: Claim contradicts the source text

### When to Reject Summaries
- Contradicted claims > 0 (hallucinations detected)
- Faithfulness < 70% (too many unsupported claims)
- Critical domain (healthcare, legal): Consider 90%+ threshold

## Production Considerations

### 1. Replace Mock Embedding
The example uses a simple character-frequency embedding for pre-filtering. In production:

```rust
// Replace MockEmbedding with real sentence-transformers
use sentence_transformers::SentenceTransformer;
let embedding_backend = Arc::new(SentenceTransformer::new("all-MiniLM-L6-v2")?);
```

### 2. Improve HTML Parsing
The example uses basic HTML tag stripping. For production:

```rust
// Use proper HTML parsing
use scraper::{Html, Selector};
let document = Html::parse_document(&html);
let selector = Selector::parse("p, h1, h2, h3, li").unwrap();
let text: String = document.select(&selector).map(|el| el.text().collect::<String>()).collect();
```

### 3. Handle Streaming
For large documents, stream the Claude response:

```rust
let stream = client.messages_stream(request).await?;
// Process chunks incrementally
```

### 4. Caching
Cache scraped content and summaries to avoid redundant API calls:

```rust
use sled::Db;
let cache: Db = sled::open("cache")?;
```

## Dependencies

This example uses:
- `spider = "2.38"` - Fast web scraping (200-1000x faster than alternatives)
- `anthropic-sdk-rust = "0.1"` - Type-safe Claude API client
- `krino = { path = "../.." }` - Groundedness verification

These are **NOT** part of the core Krino library and only live in this example workspace.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Live Summarization Pipeline               │
└─────────────────────────────────────────────────────────┘

1. Web Scraping (spider-rs)
   ├─ Input: URL
   ├─ Extract: Clean text content
   └─ Output: Source text (15k-50k chars)

2. LLM Summarization (Claude Sonnet API)
   ├─ Input: Source text + summarization prompt
   ├─ Model: claude-sonnet-4-20250514
   └─ Output: Summary (500-1000 chars)

3. Groundedness Verification (Krino NLI)
   ├─ Input: Source text + summary
   ├─ NLI Model: DeBERTa-v3-large ONNX
   ├─ Processing: Sentence-level claim decomposition
   └─ Output: Faithfulness score + per-claim verdicts

4. Decision Logic
   ├─ If faithfulness ≥ 70%: Accept summary
   ├─ If faithfulness < 70%: Flag for review
   └─ If contradictions > 0: Reject (hallucinations)
```

## Troubleshooting

### API Key Error
```
Error: ANTHROPIC_API_KEY environment variable not set
```
**Solution**: `export ANTHROPIC_API_KEY="sk-ant-..."`

### Model Not Found
```
Error: ONNX model not found at models/deberta-nli-onnx
```
**Solution**: `cd ../../scripts && uv run export_deberta_onnx.py`

### Spider Scraping Fails
```
Error: Failed to scrape any content
```
**Solutions**:
- Check URL is accessible
- Try different URL (some sites block scrapers)
- Check internet connection

### Rate Limits
If you hit Claude API rate limits, add delay between requests:
```bash
sleep 5 && cargo run -- --url "..."
```

## License

This example inherits the Krino license. See `../../LICENSE`.
