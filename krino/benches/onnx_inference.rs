//! Performance baseline benchmarks for ONNX backend.
//!
//! Measures inference latency across different input lengths to validate
//! the sub-200ms target before integrating with groundedness (SummaC matrix).
//!
//! Also benchmarks the session pool configuration (N=1 vs N=2 vs N=4) on a
//! 50-input batch representative of a typical groundedness request.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use krino::models::backends::onnx::{OnnxConfig, OnnxSequenceClassifier};
use krino::models::inference::{SequenceClassifier, SequenceClassifierInput};
use std::path::Path;

fn onnx_inference_short(c: &mut Criterion) {
    let model_path = Path::new("models/deberta-nli-onnx");
    if !model_path.join("model.onnx").exists() {
        eprintln!(
            "⚠️  Skipping benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let classifier =
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model");

    // Short pair: premise 10 words, hypothesis 10 words (expected: ~10-15ms)
    let input = SequenceClassifierInput {
        text_a: "The cat sat on the mat.".to_string(),
        text_b: "An animal was on the mat.".to_string(),
    };

    c.bench_function("onnx_inference_short", |b| {
        b.iter(|| {
            classifier
                .classify(black_box(std::slice::from_ref(&input)))
                .expect("Inference failed")
        });
    });
}

fn onnx_inference_medium(c: &mut Criterion) {
    let model_path = Path::new("models/deberta-nli-onnx");
    if !model_path.join("model.onnx").exists() {
        return;
    }

    let classifier =
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model");

    // Medium pair: premise ~50 words, hypothesis ~25 words (expected: ~15-25ms)
    let input = SequenceClassifierInput {
        text_a: "The financial advisor provided comprehensive investment recommendations covering stocks, bonds, and mutual funds. The client reviewed the portfolio suggestions carefully before making any decisions. The advisor emphasized the importance of diversification and long-term planning for retirement goals.".to_string(),
        text_b: "The advisor discussed investment options including stocks and bonds with the client.".to_string(),
    };

    c.bench_function("onnx_inference_medium", |b| {
        b.iter(|| {
            classifier
                .classify(black_box(std::slice::from_ref(&input)))
                .expect("Inference failed")
        });
    });
}

fn onnx_inference_long(c: &mut Criterion) {
    let model_path = Path::new("models/deberta-nli-onnx");
    if !model_path.join("model.onnx").exists() {
        return;
    }

    let classifier =
        OnnxSequenceClassifier::from_pretrained(model_path).expect("Failed to load ONNX model");

    // Long pair: premise ~200 words, hypothesis ~50 words (expected: ~25-50ms)
    let premise = "In the ever-evolving landscape of financial services, investment advisors face increasingly complex regulatory requirements and ethical obligations. The Securities and Exchange Commission has implemented stringent guidelines governing the provision of investment advice, particularly regarding the recommendation of specific securities to retail clients. Advisors must navigate a delicate balance between providing valuable market insights and avoiding actions that could be construed as market manipulation or unsuitable recommendations. The fiduciary duty owed to clients requires advisors to act in the best interests of their clients at all times, which includes providing full disclosure of any conflicts of interest and ensuring that recommended investments align with the client's financial goals, risk tolerance, and investment timeline. Furthermore, advisors must maintain detailed records of all client interactions and recommendations to demonstrate compliance with regulatory requirements. The complexity of modern financial products, including derivatives, structured notes, and alternative investments, requires advisors to possess deep expertise and stay current with market developments. Professional development and continuing education are essential components of maintaining the competence necessary to serve clients effectively.".to_string();

    let hypothesis = "Financial advisors must follow SEC regulations when recommending specific stocks to clients and maintain comprehensive documentation of their recommendations.".to_string();

    let input = SequenceClassifierInput {
        text_a: premise,
        text_b: hypothesis,
    };

    c.bench_function("onnx_inference_long", |b| {
        b.iter(|| {
            classifier
                .classify(black_box(std::slice::from_ref(&input)))
                .expect("Inference failed")
        });
    });
}

/// Benchmark the session pool at different vCPU budgets on a 50-input batch.
///
/// Mirrors the typical groundedness request described in NLI_PARALLELISM.md:
/// ~500 context sentences pre-filtered to top-10 per claim × 5 claims = 50 NLI inputs.
/// Session count is derived automatically from intra_op_num_threads via derive_session_pool_size.
fn onnx_inference_batch_pool(c: &mut Criterion) {
    let model_path = Path::new("models/deberta-nli-onnx");
    if !model_path.join("model.onnx").exists() {
        eprintln!(
            "⚠️  Skipping pool benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    // Build 50 representative (premise, hypothesis) pairs
    let premises = [
        "The quarterly earnings report showed a significant increase in revenue compared to the previous year.",
        "Climate change is causing more frequent and severe weather events across the globe.",
        "The new pharmaceutical drug passed all three phases of clinical trials successfully.",
        "The city council approved a budget allocation for infrastructure improvements.",
        "The research team published their findings in a peer-reviewed scientific journal.",
    ];
    let hypotheses = [
        "Company revenue grew year over year.",
        "Extreme weather events are increasing in frequency.",
        "The drug is safe for public use after clinical testing.",
        "Public funds will be used for infrastructure.",
        "Scientists shared their research results publicly.",
    ];

    let inputs: Vec<SequenceClassifierInput> = (0..50)
        .map(|i| SequenceClassifierInput {
            text_a: premises[i % premises.len()].to_string(),
            text_b: hypotheses[i % hypotheses.len()].to_string(),
        })
        .collect();

    // Drive session count via intra_op_num_threads: derive_session_pool_size maps
    // 2 vcpus → 2 sessions, 4 vcpus → 2 sessions, 8 vcpus → 4 sessions.
    let configs = [
        (
            "vcpus=1",
            OnnxConfig {
                intra_op_num_threads: 1,
                batch_size: 8,
            },
        ),
        (
            "vcpus=4",
            OnnxConfig {
                intra_op_num_threads: 4,
                batch_size: 8,
            },
        ),
        (
            "vcpus=8",
            OnnxConfig {
                intra_op_num_threads: 8,
                batch_size: 8,
            },
        ),
    ];

    let mut group = c.benchmark_group("onnx_batch_pool_50inputs");

    for (label, config) in &configs {
        let classifier =
            OnnxSequenceClassifier::from_pretrained_with_config(model_path, config.clone())
                .expect("Failed to load ONNX model");

        group.bench_with_input(BenchmarkId::new("pool", label), label, |b, _| {
            b.iter(|| {
                classifier
                    .classify(black_box(&inputs))
                    .expect("Inference failed")
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    onnx_inference_short,
    onnx_inference_medium,
    onnx_inference_long,
    onnx_inference_batch_pool,
);
criterion_main!(benches);
