use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use krino::error::Result;
use krino::models::inference::{
    TokenClassifier, TokenClassifierInput, TokenClassifierOutput, TokenPrediction,
};
use krino::modules::hallucination::{HallucinationConfig, HallucinationDetector};
use std::sync::Arc;

/// Mock token classifier for benchmarking
/// Simulates realistic hallucination detection model behavior
struct MockTokenClassifier {
    /// Simulated inference latency per batch (microseconds)
    latency_per_batch_us: u64,
    /// Fixed probability of hallucination for each token
    hallucination_prob: f64,
}

impl MockTokenClassifier {
    /// Realistic backend: ~100ms per batch
    fn realistic() -> Self {
        Self {
            latency_per_batch_us: 100_000,
            hallucination_prob: 0.1,
        }
    }

    /// Fast backend: ~50ms per batch
    fn fast() -> Self {
        Self {
            latency_per_batch_us: 50_000,
            hallucination_prob: 0.1,
        }
    }

    /// Instant backend: for measuring overhead only
    fn instant() -> Self {
        Self {
            latency_per_batch_us: 0,
            hallucination_prob: 0.1,
        }
    }
}

impl TokenClassifier for MockTokenClassifier {
    fn predict(&self, inputs: &[TokenClassifierInput]) -> Result<Vec<TokenClassifierOutput>> {
        // Simulate batch inference latency
        if self.latency_per_batch_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(
                self.latency_per_batch_us * inputs.len() as u64,
            ));
        }

        Ok(inputs
            .iter()
            .map(|input| {
                let tokens: Vec<TokenPrediction> = input
                    .input_ids
                    .iter()
                    .enumerate()
                    .map(|(idx, &token_id)| {
                        let is_hallucinated = idx % 10 == 0;
                        TokenPrediction {
                            token_id,
                            token: format!("token{idx}"),
                            predicted_class: if is_hallucinated { 1 } else { 0 },
                            probabilities: if is_hallucinated {
                                vec![1.0 - self.hallucination_prob, self.hallucination_prob]
                            } else {
                                vec![0.95, 0.05]
                            },
                            offset: input.offsets.get(idx).copied().unwrap_or((0, 0)),
                        }
                    })
                    .collect();

                TokenClassifierOutput {
                    tokens,
                    latency_ms: (self.latency_per_batch_us as f64) / 1000.0,
                }
            })
            .collect())
    }

    fn device_info(&self) -> String {
        "MockClassifier".to_string()
    }

    fn max_length(&self) -> usize {
        512
    }
}

fn mock_tokenizer() -> tokenizers::Tokenizer {
    let wp = tokenizers::models::wordpiece::WordPiece::default();
    tokenizers::Tokenizer::new(wp)
}

fn bench_short_text_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::realistic());
    let detector = HallucinationDetector::new(
        backend,
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    // ~50 tokens
    let text = "The capital of France is Paris. It is known for the Eiffel Tower \
                and excellent cuisine. Many tourists visit every year.";

    let mut group = c.benchmark_group("hallucination_realistic");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("short_50_tokens", |b| {
        b.iter(|| {
            detector.detect(black_box(text)).unwrap();
        });
    });

    group.finish();
}

fn bench_medium_text_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::realistic());
    let detector = HallucinationDetector::new(
        backend,
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    // ~200 tokens
    let text = "Artificial intelligence has made significant progress in recent years. \
                Machine learning models can now perform complex tasks like image recognition, \
                natural language processing, and game playing at superhuman levels. \
                Deep learning, a subset of machine learning, uses neural networks with \
                multiple layers to learn hierarchical representations of data. \
                The field continues to evolve rapidly with new architectures and techniques \
                being developed regularly. Applications span healthcare, finance, \
                transportation, and many other domains.";

    let mut group = c.benchmark_group("hallucination_realistic");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("medium_200_tokens", |b| {
        b.iter(|| {
            detector.detect(black_box(text)).unwrap();
        });
    });

    group.finish();
}

fn bench_long_text_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::realistic());
    let detector = HallucinationDetector::new(
        backend,
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    // ~500 tokens - requires chunking
    let text = (0..25)
        .map(|i| {
            format!(
                "This is sentence number {i}. It contains information about various topics \
                 including technology, science, and current events. The content is designed \
                 to test the hallucination detection system with realistic text."
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    let mut group = c.benchmark_group("hallucination_realistic");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("long_500_tokens", |b| {
        b.iter(|| {
            detector.detect(black_box(&text)).unwrap();
        });
    });

    group.finish();
}

fn bench_varying_lengths_fast(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::fast());

    let mut group = c.benchmark_group("hallucination_fast");

    for num_words in [25, 50, 100, 200].iter() {
        let text = (0..*num_words)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");

        let detector = HallucinationDetector::new(
            backend.clone(),
            mock_tokenizer(),
            HallucinationConfig {
                threshold: 0.5,
                max_length: 512,
                chunk_overlap: 50,
                add_special_tokens: true,
                hallucination_class: 1,
                min_span_length: 5,
            },
        );

        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_words), num_words, |b, _| {
            b.iter(|| {
                detector.detect(black_box(&text)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rag_mode_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::realistic());
    let detector = HallucinationDetector::new(
        backend,
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    let context = "Paris is the capital and most populous city of France. \
                   The city has a population of 2.2 million people.";
    let question = "What is the capital of France?";
    let answer = "Paris is the capital of France with a population of 2.2 million.";

    let mut group = c.benchmark_group("hallucination_rag");

    group.bench_function("rag_mode", |b| {
        b.iter(|| {
            detector
                .detect_rag(black_box(context), black_box(question), black_box(answer))
                .unwrap();
        });
    });

    group.finish();
}

fn bench_overhead_only(c: &mut Criterion) {
    // Zero-latency backend to measure framework overhead
    let backend = Arc::new(MockTokenClassifier::instant());
    let detector = HallucinationDetector::new(
        backend,
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    let text = "Short test text for overhead measurement.";

    let mut group = c.benchmark_group("hallucination_overhead");

    group.bench_function("framework_overhead", |b| {
        b.iter(|| {
            detector.detect(black_box(text)).unwrap();
        });
    });

    group.finish();
}

fn bench_chunking_overhead(c: &mut Criterion) {
    let backend = Arc::new(MockTokenClassifier::instant());

    let mut group = c.benchmark_group("hallucination_chunking");

    // Short text (no chunking)
    let short_text = "Short text that fits in one chunk.";
    let detector_no_chunk = HallucinationDetector::new(
        backend.clone(),
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 512,
            chunk_overlap: 50,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    group.bench_function("no_chunking", |b| {
        b.iter(|| {
            detector_no_chunk.detect(black_box(short_text)).unwrap();
        });
    });

    // Long text requiring chunking
    let long_text = (0..100)
        .map(|i| format!("This is word number {i} in a very long document."))
        .collect::<Vec<_>>()
        .join(" ");

    let detector_with_chunk = HallucinationDetector::new(
        backend.clone(),
        mock_tokenizer(),
        HallucinationConfig {
            threshold: 0.5,
            max_length: 100, // Force chunking
            chunk_overlap: 20,
            add_special_tokens: true,
            hallucination_class: 1,
            min_span_length: 5,
        },
    );

    group.bench_function("with_chunking", |b| {
        b.iter(|| {
            detector_with_chunk.detect(black_box(&long_text)).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_short_text_realistic,
    bench_medium_text_realistic,
    bench_long_text_realistic,
    bench_varying_lengths_fast,
    bench_rag_mode_realistic,
    bench_overhead_only,
    bench_chunking_overhead
);

criterion_main!(benches);
