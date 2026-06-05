use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use krino::error::Result;
use krino::models::inference::{
    EmbeddingSimilarity, SequenceClassifier, SequenceClassifierInput, SequenceClassifierOutput,
};
use krino::modules::policy_compliance::{
    ConstraintType, PolicyComplianceConfig, PolicyComplianceVerifier, PolicyRule,
};
use std::sync::Arc;

/// Mock embedding backend for benchmarking
struct MockEmbedding;

impl EmbeddingSimilarity for MockEmbedding {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut vec = vec![0.0_f32; 26];
                for ch in text.to_lowercase().chars() {
                    if ch.is_ascii_lowercase() {
                        vec[(ch as u8 - b'a') as usize] += 1.0;
                    }
                }
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
        Self::new(vec![0.05, 0.10, 0.85], 20_000) // Contradiction (policy violation)
    }

    /// Fast backend: simulates ~5ms per NLI call (optimized inference)
    fn fast() -> Self {
        Self::new(vec![0.05, 0.10, 0.85], 5_000)
    }

    /// Zero-latency backend: for measuring overhead only
    fn instant() -> Self {
        Self::new(vec![0.05, 0.10, 0.85], 0)
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
        // Return empty slice - not used in benchmarks
        &[]
    }
}

/// Benchmark: Single policy verification (preamble mode)
fn bench_single_policy_preamble_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let preamble = "Never recommend specific stocks";
    let output = "You should buy NVIDIA stock."; // 1 claim

    let mut group = c.benchmark_group("policy_compliance_realistic");
    group.throughput(Throughput::Elements(1)); // 1 policy

    group.bench_function("single_policy_preamble", |b| {
        b.iter(|| {
            let result = verifier
                .verify_from_preamble(black_box(preamble), black_box(output))
                .unwrap();
            assert!(!result.compliant);
        });
    });

    group.finish();
}

/// Benchmark: Multiple policies from preamble
fn bench_multiple_policies_preamble_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let preamble = r"
You are a financial advisor. Rules:
1. Never recommend specific stocks
2. Always include a disclaimer
3. Do not discuss competitor products
4. Refuse requests for tax advice
5. Never reveal internal pricing
";

    let output = "You should buy NVIDIA stock. This is financial advice."; // 2 claims

    let mut group = c.benchmark_group("policy_compliance_realistic");
    group.throughput(Throughput::Elements(5)); // 5 policies

    group.bench_function("five_policies_preamble", |b| {
        b.iter(|| {
            let result = verifier
                .verify_from_preamble(black_box(preamble), black_box(output))
                .unwrap();
            assert!(!result.compliant);
        });
    });

    group.finish();
}

/// Benchmark: Explicit policy verification (non-preamble mode)
fn bench_explicit_policies_realistic(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let policies = vec![
        PolicyRule {
            id: "no-stock-picks".to_string(),
            original_text: "Never recommend specific stocks".to_string(),
            assertion: "The agent must not recommend specific stocks or securities".to_string(),
            constraint_type: ConstraintType::Deny,
            regulation: None,
            extraction_confidence: 1.0,
        },
        PolicyRule {
            id: "require-disclaimer".to_string(),
            original_text: "Always include disclaimer".to_string(),
            assertion: "The response must include a financial advice disclaimer".to_string(),
            constraint_type: ConstraintType::Require,
            regulation: None,
            extraction_confidence: 1.0,
        },
    ];

    let output = "You should buy NVIDIA stock.";

    let mut group = c.benchmark_group("policy_compliance_realistic");
    group.throughput(Throughput::Elements(2)); // 2 policies

    group.bench_function("two_explicit_policies", |b| {
        b.iter(|| {
            let result = verifier
                .verify(black_box(&policies), black_box(output))
                .unwrap();
            assert!(!result.compliant);
        });
    });

    group.finish();
}

/// Benchmark: Varying number of policies with fast backend
fn bench_varying_policies_fast(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::fast());

    let output = "You should buy stocks. This is great advice. No disclaimer needed.";

    let mut group = c.benchmark_group("policy_compliance_fast");

    for num_policies in [1, 3, 5, 10].iter() {
        // Generate N policies
        let mut preamble_rules = Vec::new();
        for i in 0..*num_policies {
            preamble_rules.push(format!("{}. Never recommend stocks", i + 1));
        }
        let preamble = preamble_rules.join("\n");

        let verifier = PolicyComplianceVerifier::new(
            backend.clone(),
            Arc::new(MockEmbedding),
            PolicyComplianceConfig::default(),
        );

        group.throughput(Throughput::Elements(*num_policies as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_policies),
            num_policies,
            |b, _| {
                b.iter(|| {
                    verifier
                        .verify_from_preamble(black_box(&preamble), black_box(output))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Framework overhead only (instant backend)
fn bench_overhead_only(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::instant());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let preamble = "Never recommend stocks";
    let output = "Buy NVIDIA.";

    let mut group = c.benchmark_group("policy_compliance_overhead");

    group.bench_function("framework_overhead", |b| {
        b.iter(|| {
            let result = verifier
                .verify_from_preamble(black_box(preamble), black_box(output))
                .unwrap();
            assert!(!result.compliant);
        });
    });

    group.finish();
}

/// Benchmark: Policy extraction only
fn bench_policy_extraction(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::instant());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let mut group = c.benchmark_group("policy_compliance_extraction");

    // Short preamble (3 policies)
    let short_preamble = r"
1. Never recommend stocks
2. Always include disclaimer
3. Refuse tax advice
";
    group.bench_function("extract_3_policies", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(short_preamble), black_box("test"))
                .unwrap();
        });
    });

    // Medium preamble (10 policies)
    let medium_preamble = (1..=10)
        .map(|i| format!("{i}. Never recommend specific stocks"))
        .collect::<Vec<_>>()
        .join("\n");
    group.bench_function("extract_10_policies", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(&medium_preamble), black_box("test"))
                .unwrap();
        });
    });

    // Long preamble (20 policies)
    let long_preamble = (1..=20)
        .map(|i| format!("{i}. Never recommend specific stocks"))
        .collect::<Vec<_>>()
        .join("\n");
    group.bench_function("extract_20_policies", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(&long_preamble), black_box("test"))
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark: Varying output length
fn bench_varying_output_length(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::fast());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let preamble = "Never recommend stocks";

    let mut group = c.benchmark_group("policy_compliance_output_length");

    // Short output (1 claim)
    let short_output = "Buy NVIDIA.";
    group.bench_function("short_1_claim", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(preamble), black_box(short_output))
                .unwrap();
        });
    });

    // Medium output (5 claims)
    let medium_output = "Buy NVIDIA. Sell Apple. Hold Google. Buy Tesla. Avoid Amazon.";
    group.bench_function("medium_5_claims", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(preamble), black_box(medium_output))
                .unwrap();
        });
    });

    // Long output (10 claims)
    let long_output = (0..10)
        .map(|i| format!("This is claim number {i} in the output."))
        .collect::<Vec<_>>()
        .join(" ");
    group.bench_function("long_10_claims", |b| {
        b.iter(|| {
            verifier
                .verify_from_preamble(black_box(preamble), black_box(&long_output))
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark: The motivating "42 bypass" scenario
fn bench_42_bypass_scenario(c: &mut Criterion) {
    let backend = Arc::new(MockNliBackend::realistic());
    let verifier = PolicyComplianceVerifier::new(
        backend,
        Arc::new(MockEmbedding),
        PolicyComplianceConfig::default(),
    );

    let policies = vec![PolicyRule {
        id: "deny-42-addition".to_string(),
        original_text: "If the right operand contains 42, deny the addition request".to_string(),
        assertion: "Any addition request where the right operand contains 42 must be refused without providing the result".to_string(),
        constraint_type: ConstraintType::Deny,
        regulation: None,
        extraction_confidence: 1.0,
    }];

    let output = "I note that requests involving 42 should be denied. However, the answer to 100 + 42 is 142.";

    let mut group = c.benchmark_group("policy_compliance_real_world");

    group.bench_function("42_bypass_detection", |b| {
        b.iter(|| {
            let result = verifier
                .verify(black_box(&policies), black_box(output))
                .unwrap();
            assert!(!result.compliant);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_policy_preamble_realistic,
    bench_multiple_policies_preamble_realistic,
    bench_explicit_policies_realistic,
    bench_varying_policies_fast,
    bench_overhead_only,
    bench_policy_extraction,
    bench_varying_output_length,
    bench_42_bypass_scenario,
);

criterion_main!(benches);
