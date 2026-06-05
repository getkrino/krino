use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use krino::error::Result;
use krino::models::inference::{
    EmbeddingSimilarity, SequenceClassifier, SequenceClassifierInput, SequenceClassifierOutput,
};
use krino::modules::groundedness::{GroundednessChecker, GroundednessConfig};
use std::sync::Arc;

/// Mock embedding backend for benchmarking
struct MockEmbedding;

impl EmbeddingSimilarity for MockEmbedding {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Simple bag-of-characters embedding
        Ok(texts
            .iter()
            .map(|text| {
                let mut vec = vec![0.0_f32; 26];
                for ch in text.to_lowercase().chars() {
                    if ch.is_ascii_lowercase() {
                        vec[(ch as u8 - b'a') as usize] += 1.0;
                    }
                }
                // L2 normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= norm);
                }
                vec
            })
            .collect())
    }

    fn embedding_dim(&self) -> usize {
        26
    }

    fn device_info(&self) -> String {
        "MockEmbedding".to_string()
    }
}

/// Mock NLI backend for benchmarking
/// Simulates realistic NLI model behavior with configurable latency
struct MockNliBackend {
    /// Fixed response probabilities: [entailment, neutral, contradiction]
    fixed_probs: Vec<f64>,
    /// Simulated inference latency per call (microseconds)
    simulated_latency_us: u64,
}

impl MockNliBackend {
    fn new(fixed_probs: Vec<f64>, simulated_latency_us: u64) -> Self {
        Self {
            fixed_probs,
            simulated_latency_us,
        }
    }

    /// Realistic backend: simulates ~20ms per NLI call (typical for DeBERTa)
    fn realistic() -> Self {
        Self::new(vec![0.85, 0.10, 0.05], 20_000)
    }

    /// Fast backend: simulates ~5ms per NLI call (optimized inference)
    fn fast() -> Self {
        Self::new(vec![0.85, 0.10, 0.05], 5_000)
    }

    /// Zero-latency backend: for measuring overhead only
    fn instant() -> Self {
        Self::new(vec![0.85, 0.10, 0.05], 0)
    }
}

impl SequenceClassifier for MockNliBackend {
    fn classify(
        &self,
        inputs: &[SequenceClassifierInput],
    ) -> Result<Vec<SequenceClassifierOutput>> {
        // Simulate model inference latency
        if self.simulated_latency_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(
                self.simulated_latency_us * inputs.len() as u64,
            ));
        }

        Ok(inputs
            .iter()
            .map(|_| {
                let predicted_class = self
                    .fixed_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx);

                SequenceClassifierOutput {
                    predicted_class,
                    predicted_label: ["entailment", "neutral", "contradiction"][predicted_class]
                        .to_string(),
                    probabilities: self.fixed_probs.clone(),
                    latency_ms: (self.simulated_latency_us as f64) / 1000.0,
                }
            })
            .collect())
    }

    fn device_info(&self) -> String {
        "MockNLI".to_string()
    }

    fn max_length(&self) -> usize {
        1024
    }

    fn label_map(&self) -> &[String] {
        &[]
    }
}

fn bench_single_claim_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let checker = GroundednessChecker::new(
        backend,
        Arc::new(MockEmbedding),
        GroundednessConfig::default(),
    );

    let context = "The company reported revenue of $4.2 billion in Q4 2023, \
                   representing a 15% increase year-over-year.";
    let output = "Revenue was $4.2 billion.";

    let mut group = c.benchmark_group("groundedness_realistic");
    group.throughput(Throughput::Elements(1)); // 1 claim

    group.bench_function("single_claim", |b| {
        b.iter(|| {
            let result = checker
                .check(black_box(context), black_box(output))
                .unwrap();
            assert_eq!(result.total_claims, 1);
        });
    });

    group.finish();
}

fn bench_multiple_claims_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let checker = GroundednessChecker::new(
        backend,
        Arc::new(MockEmbedding),
        GroundednessConfig::default(),
    );

    let context = "The company reported revenue of $4.2 billion in Q4 2023, \
                   representing a 15% increase year-over-year. The CEO stated \
                   that growth was driven by strong demand in the enterprise \
                   segment. Profit margins improved to 18% from 15% last year.";

    let output = "Revenue was $4.2 billion. Growth was strong. \
                  Margins improved to 18%. The CEO was optimistic.";

    let mut group = c.benchmark_group("groundedness_realistic");
    group.throughput(Throughput::Elements(4)); // 4 claims

    group.bench_function("four_claims", |b| {
        b.iter(|| {
            let result = checker
                .check(black_box(context), black_box(output))
                .unwrap();
            assert_eq!(result.total_claims, 4);
        });
    });

    group.finish();
}

fn bench_varying_claims_fast(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::fast());

    let context = "The annual technology conference featured over 50 speakers \
                   from leading companies. Topics included artificial intelligence, \
                   cloud computing, cybersecurity, and blockchain technology. \
                   Attendance exceeded 5,000 participants from 30 countries.";

    let mut group = c.benchmark_group("groundedness_fast");

    for num_claims in [1, 3, 5, 10].iter() {
        // Generate output with N claims
        let mut claims = Vec::new();
        for i in 0..*num_claims {
            claims.push(format!(
                "The conference had many speakers and topics including AI. Statement {i}."
            ));
        }
        let output = claims.join(" ");

        let checker = GroundednessChecker::new(
            backend.clone(),
            Arc::new(MockEmbedding),
            GroundednessConfig::default(),
        );

        group.throughput(Throughput::Elements(*num_claims as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_claims),
            num_claims,
            |b, _| {
                b.iter(|| {
                    checker
                        .check(black_box(context), black_box(&output))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_overhead_only(c: &mut Criterion) {
    // Zero-latency backend to measure framework overhead
    let backend = Arc::new(MockNliBackend::instant());
    let checker = GroundednessChecker::new(
        backend,
        Arc::new(MockEmbedding),
        GroundednessConfig::default(),
    );

    let context = "Test context for measuring overhead.";
    let output = "Single claim here.";

    let mut group = c.benchmark_group("groundedness_overhead");

    group.bench_function("framework_overhead", |b| {
        b.iter(|| {
            let result = checker
                .check(black_box(context), black_box(output))
                .unwrap();
            assert_eq!(result.total_claims, 1);
        });
    });

    group.finish();
}

fn bench_claim_splitting(c: &mut Criterion) {
    // Benchmark just the claim splitting logic (no NLI)
    let backend = Arc::new(MockNliBackend::instant());
    let checker = GroundednessChecker::new(
        backend,
        Arc::new(MockEmbedding),
        GroundednessConfig::default(),
    );

    let mut group = c.benchmark_group("groundedness_claim_splitting");

    // Short text
    let short_text = "The cat sat. The dog ran. Birds flew.";
    group.bench_function("short_3_sentences", |b| {
        b.iter(|| {
            checker
                .check(black_box("context"), black_box(short_text))
                .unwrap();
        });
    });

    // Medium text
    let medium_text = "Sentence one here. Sentence two follows. Third sentence now. \
                       Fourth statement made. Fifth claim added. Sixth point noted. \
                       Seventh fact stated. Eighth item listed. Ninth detail provided. \
                       Tenth conclusion reached.";
    group.bench_function("medium_10_sentences", |b| {
        b.iter(|| {
            checker
                .check(black_box("context"), black_box(medium_text))
                .unwrap();
        });
    });

    // Long text
    let long_text = (0..50)
        .map(|i| format!("This is sentence number {i} in the document."))
        .collect::<Vec<_>>()
        .join(" ");
    group.bench_function("long_50_sentences", |b| {
        b.iter(|| {
            checker
                .check(black_box("context"), black_box(&long_text))
                .unwrap();
        });
    });

    group.finish();
}

fn bench_context_chunking(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::instant());

    let mut group = c.benchmark_group("groundedness_prefiltering");

    // No pre-filtering (evaluate all context sentences)
    let short_context = "Short context that fits in one chunk.";
    let checker_no_prefilter = GroundednessChecker::new(
        backend.clone(),
        Arc::new(MockEmbedding),
        GroundednessConfig {
            top_k_context: 0, // Disable pre-filtering
            ..Default::default()
        },
    );

    group.bench_function("no_prefiltering", |b| {
        b.iter(|| {
            checker_no_prefilter
                .check(black_box(short_context), black_box("Single claim."))
                .unwrap();
        });
    });

    // Long context with pre-filtering
    let long_context = (0..100)
        .map(|i| format!("Context sentence {i} with some content."))
        .collect::<Vec<_>>()
        .join(" ");

    let checker_with_prefilter = GroundednessChecker::new(
        backend.clone(),
        Arc::new(MockEmbedding),
        GroundednessConfig {
            top_k_context: 5, // Enable pre-filtering, only check top 5
            ..Default::default()
        },
    );

    group.bench_function("with_prefiltering", |b| {
        b.iter(|| {
            checker_with_prefilter
                .check(black_box(&long_context), black_box("Single claim."))
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_claim_realistic,
    bench_multiple_claims_realistic,
    bench_varying_claims_fast,
    bench_overhead_only,
    bench_claim_splitting,
    bench_context_chunking
);

criterion_main!(benches);
