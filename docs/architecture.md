# Architecture

How Krino decides whether an LLM output is faithful to a context.

This is the mental model. The authoritative description is the engine
source under `krino/src/modules/groundedness.rs`; this document
should track it but lags behind by definition.

## The big picture

A single call to `POST /api/v1/evaluate` flows through:

```
┌─────────────┐  context + output
│   HTTP      │ ───────────────────┐
│  handler    │                    │
└─────────────┘                    ▼
                          ┌─────────────────┐
                          │ Worker pool     │   (async-channel + Rayon)
                          │ (1+ workers)    │
                          └────────┬────────┘
                                   ▼
┌─────────────────────────────────────────────────────────┐
│  GroundednessChecker::check_with_overrides              │
│                                                         │
│  1. split context + output into sentences               │
│  2. substring fast-path: skip claims that appear        │
│     verbatim in context                                 │
│  3. embedding pre-filter: top-K candidate sentences     │
│     per claim by cosine similarity                      │
│  4. NLI batch inference: (claim, candidate) → probs     │
│  5. per-claim verdict assembly                          │
│     · entailment / contradiction / neutral              │
│     · partial (compound claims with multi-evidence)     │
│  6. aggregate into faithfulness score                   │
└─────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  EvaluateResp.  │
                          │  (JSON)         │
                          └─────────────────┘
```

The whole pipeline is **deterministic by construction**. Same inputs,
same outputs. No randomness, no LLM-as-judge.

## Stage 1: sentence splitting

`split_into_claims` is rule-based. It splits on `.`, `!`, `?`, and
`;`, with special-case handling for:

- Common abbreviations (`Mr.`, `Dr.`, `etc.`, `i.e.`, `U.S.`)
- Decimal numbers (`3.14` is not a boundary)
- Semicolons as clause separators

It's not a model. It's deterministic by definition. Splitting time is
~1ms on typical inputs and never the bottleneck.

The context and the output both go through the same splitter, so the
two "sentence" populations are produced identically. The output's
splits become *claims* to verify. The context's splits become
*candidates* to verify against.

## Stage 2: substring fast-path

Some claims appear verbatim in the context — particularly common when
the LLM is summarizing and quotes a sentence back. For those, NLI is
overkill: substring containment is sufficient evidence of entailment.

`normalize_for_substring` lowercases, collapses whitespace, and strips
trailing punctuation. Then for each claim:

1. Normalize the claim.
2. Check if any normalized context sentence contains it as a substring.
3. If yes → verdict is `entailment`, evidence is the matching sentence,
   entailment probability is set to 1.0, and the claim never sees NLI.

Claims under 12 characters skip the fast-path; they false-positive too
easily ("the" appears everywhere). Anything longer is unambiguous
enough.

The fast-path is purely an optimization — it changes nothing about
verdict quality on the claims it catches, since substring containment
is a stricter proof of entailment than the NLI model could provide.

## Stage 3: embedding pre-filter

For claims that survive the fast-path, the engine doesn't run NLI on
the full Cartesian product of `claims × context_sentences` — that's
prohibitive for any context bigger than a few sentences.

Instead, both populations are embedded using
`sentence-transformers/all-MiniLM-L6-v2` (quantized ONNX). The engine
computes the full claim × context cosine-similarity matrix, then for
each claim keeps the top-K most-similar context sentences. K defaults
to 10.

The pre-filter is a **recall** mechanism, not a verdict mechanism. Its
job is to make sure the *correct* evidence sentence is in the
candidate set for NLI to evaluate, not to decide whether the claim is
supported. NLI does the deciding.

The pre-filter can be disabled per-request with
`config.top_k_context: 0`, which sends every `(claim, context_sentence)`
pair to NLI. Useful for audit probes and tiny contexts.

## Stage 4: NLI batch inference

For each `(claim, candidate)` pair, run the NLI model. The default
model is RoBERTa-large-MNLI in static INT8 quantization, hosted as
ONNX. The model outputs three probabilities per pair:

- `entailment` — premise entails hypothesis
- `neutral` — no relation
- `contradiction` — premise denies hypothesis

The engine batches pairs into ORT forward passes of `batch_size` at a
time (16 by default). The batch dimension is parallel; the per-batch
forward pass uses `threads_per_worker` ORT intra-op threads.

The model is the heavy cost: ~120ms per batch on c8a.2xlarge for
typical sequence lengths. NLI dominates the latency budget on any
non-trivial input.

## Stage 5: per-claim verdict

For each claim, the engine looks at the set of `(candidate, probs,
similarity)` tuples it has and decides a verdict.

### Single-best-evidence path

By default, the verdict is the one for the single best evidence:

1. Find the candidate with the highest `max(entailment, contradiction)`
   — the most informative signal. (Neutral is never the most
   informative.)
2. If that candidate's `contradiction ≥ contradiction_threshold`
   (default 0.5) and dominates the other classes → verdict is
   `contradiction`.
3. Else if `entailment` is the max → verdict is `entailment`.
4. Else → verdict is `neutral`.

This works well for atomic claims (one fact per sentence) that have
clear matching sentences in context.

### Multi-evidence aggregation (the `partial` path)

It fails on **compound claims** — multi-fact sentences like:

> Rust is used in web services (AWS, Cloudflare, npm), operating
> systems (Linux kernel, Windows, Android), and browsers (Firefox's
> Gecko engine).

No single context sentence covers all three domains. Each candidate
sentence individually scores ~0.27 entailment / ~0.59 neutral, so the
single-best-evidence path lands on `neutral` — even though the union
of three different sentences clearly supports the claim.

The engine detects this case:

1. `is_compound_claim` flags the claim as compound (it contains
   conjunction patterns: `", and "`, em-dashes, semicolons, etc.).
2. If the single-best-evidence verdict is `neutral` AND no candidate
   exceeded `contradiction_threshold`, the engine runs
   `collect_partial_evidence` over the candidate set.
3. A candidate qualifies as partial evidence iff *all* hold:
   - `entailment ≥ partial_threshold` (default 0.2, a noise floor)
   - `entailment > contradiction` (the model leans toward support)
   - `neutral ≤ partial_neutral_ceiling` (default 0.65 — the model is
     *interestedly* uncertain, not dismissing the pair as off-topic)
   - `similarity ≥ partial_similarity_floor` (default 0.7, when
     pre-filter ran)
4. If 2+ distinct candidates qualify → verdict is `partial`, supported
   is true, the claim's headline `entailment_prob` is the mean of
   contributing entailments, and the candidates are returned in
   `supporting_evidence`.

The three-condition rule was calibrated against probe data on RoBERTa-
large-MNLI INT8: true compound claims showed `e ≈ 0.27 / n ≈ 0.59 /
sim ≈ 0.79`, unrelated content showed `e ≈ 0.19 / n ≈ 0.71 / sim ≈
0.51`. Entailment alone gives a 7-point gap (fragile); the
conjunction of entailment, neutral, and similarity gives a much
sharper boundary. Defaults assume this model; other backends need
recalibration.

### What partial intentionally is not

Partial is **not** sub-claim decomposition. The engine doesn't try to
split the compound claim into parts and verify each. It reuses the
existing matrix and reframes the verdict as "the union of these
sentences supports the union of facts in the claim." It's a heuristic
over NLI outputs, not compositional entailment. Two sentences each
weakly entailing the same half of a compound claim can trip the rule
— surfacing the multi-evidence list lets the caller catch this.

## Stage 6: aggregation

The overall `score` is `supported_claims / total_claims` where
"supported" means verdict `entailment` or `partial` (or `neutral` if
`treat_neutral_as_unsupported` is false, the default).

`engine_confidence` is the fraction of claims with a *decisive*
verdict (entailment, contradiction, or partial — anything except
neutral). When it's low, the headline score is shaky and the caller
should look at the per-claim verdicts.

## Threading and concurrency

Three thread populations:

- **Tokio runtime** — handles HTTP I/O and request decoding. Typical
  axum default.
- **Worker OS threads** — one per `n_workers`, created with
  `std::thread::spawn`. Each blocking-reads from an `async_channel`,
  installs its work into a private Rayon pool, and runs the engine.
- **Per-worker Rayon threads** — `threads_per_worker` of them per
  worker. These provide the parallelism for the embedding and NLI
  passes. ORT uses them as its intra-op thread pool.

The default configuration (1 worker × (vCPUs − 1) threads) maximizes
single-request latency. The latency-throughput tradeoff is discussed
in [configuration.md](configuration.md#workers).

## Determinism

The contract: same `(context, output, config)` produces same
`EvaluateResponse`. Things that could break determinism, and how the
engine protects against them:

- **HashMap iteration order.** The engine uses `HashMap` for evidence
  bookkeeping but never iterates one to produce output; results are
  sorted by `claim_idx` before serialization.
- **Floating-point ordering.** `partial_cmp` on `f32`/`f64` is used
  with stable fallbacks (`Ordering::Equal` on NaN) and stable sorts.
- **ONNX Runtime.** ORT is deterministic for a given build of the
  underlying libonnxruntime; we pin the version and use the `ort`
  crate's `download-binaries` feature to get reproducible binaries.
- **Embedding model.** all-MiniLM-L6-v2 is deterministic; INT8
  quantization is deterministic; the engine reads a single ONNX
  session per worker.

A determinism regression is treated as a bug, not a tuning question.

## What's deliberately not here

- **No LLM-as-judge.** Krino's verdicts come from NLI models — small,
  task-specific classifiers, not chat models. This is the founding
  design constraint.
- **No manual NLP heuristics for decision-making.** The compound-claim
  detector uses a small list of conjunction indicators for the
  *flagging* step (false positives just decorate the response), but
  verdicts always come from the model.
- **No streaming or partial responses.** Each `/evaluate` is a single
  request/response.
- **No GPU path yet.** All inference is CPU-only ONNX with INT8
  quantization. CUDA/Metal backends exist in Candle but aren't wired
  through the API today.
