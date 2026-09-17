#!/usr/bin/env python3
"""Generate the local 5K-doc corpus + 60-query golden set.

WHY SYNTHETIC: the design targets the Kaggle arXiv snapshot (~4GB download);
on a constrained night build that download is the risky path, so the repo
ships a deterministic synthetic corpus of realistic CS-paper titles/abstracts
(seeded — same output every run) and keeps the real loader in
scripts/fetch_arxiv.py for later. Everything downstream (ingest, hybrid
retrieval, evals, SSE API) runs identically on either corpus.

Usage: python3 scripts/gen_corpus.py [--n 5000] [--golden 60]
Outputs: data/corpus_5k.jsonl, data/golden_60.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOPICS = {
    "transformer efficiency": {
        "methods": ["sparse attention", "linear attention", "FlashAttention kernels",
                    "low-rank attention approximation", "token pruning",
                    "adaptive computation time", "depthwise-separated feedforward blocks"],
        "tasks": ["long-document language modeling", "machine translation inference",
                  "speech transcription", "vision transformer training"],
        "datasets": ["Wiki-40B", "LibriSpeech", "ImageNet-21k", "WMT23"],
        "metrics": ["perplexity", "tokens per second", "GPU memory footprint",
                    "training FLOPs"],
    },
    "retrieval-augmented generation": {
        "methods": ["dense passage retrieval", "reciprocal rank fusion",
                    "cross-encoder reranking", "query expansion", "hybrid BM25 fusion",
                    "chunk boundary optimization", "contextual chunk headers"],
        "tasks": ["open-domain question answering", "enterprise knowledge search",
                  "multi-hop reasoning", "citation generation"],
        "datasets": ["Natural Questions", "HotpotQA", "MS MARCO", "arXiv abstracts"],
        "metrics": ["hit@10", "MRR", "faithfulness", "answer relevancy"],
    },
    "state space models": {
        "methods": ["selective scan kernels", "input-dependent transitions",
                    "parallel associative scans", "hybrid convolution-SSM blocks",
                    "gated state expansions"],
        "tasks": ["long-range sequence modeling", "audio generation",
                  "genomic sequence analysis", "language modeling at scale"],
        "datasets": ["Long Range Arena", "enwik8", "GenomeNGS", "The Pile"],
        "metrics": ["bits per byte", "forward-backward wall-clock", "associative recall"],
    },
    "diffusion models": {
        "methods": ["classifier-free guidance", "latent diffusion denoising",
                    "flow matching", "consistency distillation", "noise schedule tuning"],
        "tasks": ["text-to-image synthesis", "video generation", "molecular conformer generation"],
        "datasets": ["LAION-2B", "UCF-101", "QM9"],
        "metrics": ["FID score", "clip similarity", "sampling steps per image"],
    },
    "reinforcement learning from human feedback": {
        "methods": ["proximal policy optimization on preference pairs",
                    "direct preference optimization", "reward model ensembles",
                    "iterated amplification", "constitutional self-critique"],
        "tasks": ["alignment of instruction-following agents", "summarization quality",
                  "harmlessness tuning", "reasoning-chain evaluation"],
        "datasets": ["Anthropic HH-RLHF", "OpenAssistant conversations", "MT-Bench prompts"],
        "metrics": ["win rate against reference", "reward-model accuracy", "KL divergence"],
    },
    "mixture of experts": {
        "methods": ["top-k router gating", "expert capacity balancing",
                    "load-balancing auxiliary losses", "expert parallelism sharding"],
        "tasks": ["scaling language models sparsely", "multilingual routing",
                  "code generation inference"],
        "datasets": ["The Pile", "ROOTS", "The Stack"],
        "metrics": ["quality per training FLOP", "router load entropy", "expert utilization"],
    },
    "vector databases": {
        "methods": ["HNSW graph indexing", "product quantization", "int8 scalar quantization",
                    "distributed segment compaction", "filtered approximate search"],
        "tasks": ["billion-scale nearest neighbor search", "hybrid sparse-dense retrieval",
                  "streaming index updates"],
        "datasets": ["SIFT1B", "GloVe", "MS MARCO passages"],
        "metrics": ["recall@100 vs queries per second", "index build hours", "RAM per vector"],
    },
    "federated learning": {
        "methods": ["FedProx heterogeneity regularization", "secure aggregation protocols",
                    "client drift correction", "personalization layers"],
        "tasks": ["cross-device model training", "hospital network cooperation",
                  "keyboard next-word prediction"],
        "datasets": ["FEMNIST", "StackOverflow federated", "MIMIC-III splits"],
        "metrics": ["communication rounds to target accuracy", "client retention", "privacy budget"],
    },
    "model quantization": {
        "methods": ["post-training int8 quantization", "GPTQ weight-only compression",
                    "quantization-aware training", "mixed-precision matmul kernels"],
        "tasks": ["on-device inference", "batch serving throughput", "edge deployment"],
        "datasets": ["C4", "Common Sense QA", "SantaCoder eval suite"],
        "metrics": ["zero-shot accuracy gap", "model size in GB", "tokens per second per watt"],
    },
    "knowledge graphs": {
        "methods": ["embedding-based link prediction", "relation-aware message passing",
                    "LLM-guided schema induction", "entity disambiguation pipelines"],
        "tasks": ["biomedical relation extraction", "question answering over KGs",
                  "entity alignment across corpora"],
        "datasets": ["Wikidata5M", "FB15k-237", "UMLS"],
        "metrics": ["hits@10", "mean reciprocal rank", "F1 entity linking"],
    },
    "llm agents": {
        "methods": ["tool-augmented planning", "self-reflection loops",
                    "ReAct-style interleaved reasoning", "memory consolidation"],
        "tasks": ["autonomous web navigation", "multi-step code repair",
                  "data analysis copilots", "benchmark task completion"],
        "datasets": ["WebArena", "SWE-bench", "GAIA"],
        "metrics": ["task success rate", "steps per solve", "cost per episode"],
    },
    "time series forecasting": {
        "methods": ["patch-based transformer backbones", "frequency-domain mixing",
                    "foundation-model zero-shot forecasting", "covariate-aware attention"],
        "tasks": ["electricity load prediction", "cloud resource scaling",
                  "retail demand forecasting"],
        "datasets": ["ETT", "Electricity Load Diagrams", "M4 competition"],
        "metrics": ["MASE", "sMAPE", "CRPS"],
    },
    "anomaly detection": {
        "methods": ["autoencoder reconstruction error", "isolation forests at scale",
                    "contrastive normality learning", "spectral residual baselines"],
        "tasks": ["log anomaly detection", "metric drift alerting", "network intrusion detection"],
        "datasets": ["HDFS logs", "KDD Cup 99", "Yahoo Webscope S5"],
        "metrics": ["precision at recall 0.95", "F1", "mean time to detect"],
    },
    "data pipelines": {
        "methods": ["declarative transformation graphs", "incremental view maintenance",
                    "schema-on-read validation", "exactly-once sink semantics"],
        "tasks": ["lakehouse ETL reliability", "late-arriving data handling",
                  "pipeline cost optimization"],
        "datasets": ["TPC-DS 10TB", "internal clickstream samples", "NYC taxi trips"],
        "metrics": ["dollars per TB processed", "p99 pipeline latency", "freshness lag"],
    },
    "streaming systems": {
        "methods": ["watermark-based windowing", "exactly-once state backends",
                    "adaptive backpressure", "two-phase commit sources"],
        "tasks": ["real-time feature serving", "fraud detection at the edge",
                  "event-time joins"],
        "datasets": ["Nexmark", "Kafka benchmark traces", "DEBS 2014 calls"],
        "metrics": ["events per second per core", "state size", "checkpoint duration"],
    },
    "distributed training": {
        "methods": ["ZeRO-style sharding", "tensor-parallel pipeline hybrid",
                    "gradient compression", "asynchronous optimizer steps"],
        "tasks": ["pretraining 70B models", "multi-cloud bursting", "fault-tolerant resumption"],
        "datasets": ["RedPajama", "RefinedWeb", "OpenWebText2"],
        "metrics": ["model FLOPs utilization", "mean time between failures tolerated",
                    "steps per hour"],
    },
    "program synthesis": {
        "methods": ["execution-guided search", "library learning from demonstrations",
                    "spec-driven decoding", "test-time repair loops"],
        "tasks": ["spreadsheet formula synthesis", "data wrangling script generation",
                  "SQL from natural language"],
        "datasets": ["Spider", "DS-1000", "Excel benchmarks"],
        "metrics": ["exact match accuracy", "pass@k", "compilation rate"],
    },
    "adversarial robustness": {
        "methods": ["adversarial training with strong attacks", "randomized smoothing",
                    "certified radius training", "input sanitization layers"],
        "tasks": ["image classifier hardening", "LLM prompt injection defense",
                  "speech spoofing detection"],
        "datasets": ["CIFAR-10-C", "AdvBench", "ASVspoof 2019"],
        "metrics": ["robust accuracy under attack", "certified accuracy", "clean accuracy drop"],
    },
    "differential privacy": {
        "methods": ["DP-SGD gradient clipping", "privacy accounting amplification",
                    "local differential privacy mechanisms", "federated central aggregation"],
        "tasks": ["private language model fine-tuning", "telemetry aggregation", "census data release"],
        "datasets": ["Stack Overflow DP", "ACS income", "synthetic streams"],
        "metrics": ["epsilon-delta budget", "utility at fixed epsilon", "training slowdown"],
    },
    "recommender systems": {
        "methods": ["two-tower retrieval models", "sequential recommendation transformers",
                    "contrastive negative sampling", "debiasing inverse propensity scoring"],
        "tasks": ["large catalog retrieval", "session-based recommendation", "cold-start onboarding"],
        "datasets": ["MovieLens-25M", "Amazon Reviews", "KuaiRand"],
        "metrics": ["recall@20", "normalized discounted cumulative gain", "coverage"],
    },
    "causal inference": {
        "methods": ["double machine learning", "propensity score matching at scale",
                    "synthetic control baselines", "front-door adjustment with text"],
        "tasks": ["uplift modeling in marketing", "observational treatment effect estimation"],
        "datasets": ["IHDP", "Jobs", "Crite-Uplift"],
        "metrics": ["ATE error", "policy value", "precision in estimation of heterogeneous effects"],
    },
    "weak supervision": {
        "methods": ["labeling function aggregation", "generative label models",
                    "snippet-level distant supervision", "conflict-aware training"],
        "tasks": ["training set bootstrapping", "noisy label cleaning", "relation dataset creation"],
        "datasets": ["Snorkel benchmarks", "TACRED distant", "CheXpert labels"],
        "metrics": ["label F1 vs gold", "downstream model delta", "labeling hours saved"],
    },
    "graph neural networks": {
        "methods": ["message passing with edge features", "graph transformer attention",
                    "subgraph sampling training", "over-squashing mitigation"],
        "tasks": ["molecular property prediction", "fraud ring detection", "traffic forecasting"],
        "datasets": ["ogbg-molpcba", "Elliptic", "METR-LA"],
        "metrics": ["average precision", "ROC-AUC", "MAE"],
    },
    "active learning": {
        "methods": ["uncertainty sampling with ensembles", "diversity-aware acquisition",
                    "expected gradient length", "human-in-the-loop batch selection"],
        "tasks": ["annotation budget optimization", "fine-tuning data selection",
                  "out-of-distribution discovery"],
        "datasets": ["CIFAR-100 pools", "IMDB review pools", "medical imaging queues"],
        "metrics": ["accuracy per labeled example", "area under the budget curve"],
    },
    "multimodal learning": {
        "methods": ["contrastive image-text pretraining", "late-fusion adapters",
                    "audio-visual synchronization losses", "visual instruction tuning"],
        "tasks": ["image captioning", "video question answering", "document understanding"],
        "datasets": ["LAION-COCO", "ActivityNet-QA", "DocVQA"],
        "metrics": ["CIDEr", "accuracy", "exact match"],
    },
    "neural code generation": {
        "methods": ["repository-level context retrieval", "fill-in-the-middle objectives",
                    "execution-feedback fine-tuning", "unit-test reward shaping"],
        "tasks": ["function completion", "bug fixing from stack traces", "code migration"],
        "datasets": ["HumanEval-X", "Defects4J", "CrossCodeEval"],
        "metrics": ["pass@1", "patch applicability rate", "edit similarity"],
    },
}

TITLE_PATTERNS = [
    "{Method}: {Adj} {TopicNoun} for {Task}",
    "{AdjCap} {TopicNoun} via {Method}",
    "Rethinking {TopicNoun}: {Method} at {Scale}",
    "{Method} for {Task}: A {Adj} Study",
    "On the {Adjity} of {TopicNoun} in {Task}",
    "Scaling {TopicNoun} with {Method}",
    "{Method}: Making {Task} {Adj2}",
    "A {Adj} Analysis of {Method} for {TopicNoun}",
]

ADJS = ["scalable", "robust", "efficient", "principled", "adaptive", "sparsely activated",
        "memory-bounded", "low-latency", "cost-aware", "self-improving"]
ADJITY = ["scalability", "robustness", "efficiency", "limitations", "trade-offs"]
ADJ2 = ["practical", "fast", "reliable", "affordable"]
SCALES = ["10B scale", "the trillion-token regime", "production traffic",
          "the edge", "low-resource settings", "the data-center budget"]

PROBLEM = [
    "{TopicNoun} remains a bottleneck for {task}: existing approaches struggle with {issue}.",
    "Despite progress in {TopicNoun}, {task} still suffers from {issue} in practice.",
    "{task} at {scale} exposes fundamental limits in current {TopicNoun} methods, notably {issue}.",
]
ISSUES = ["quadratic cost", "hallucinated citations", "catastrophic drift",
          "cold-start sparsity", "distribution shift", "memory pressure",
          "label noise", "latency budgets", "sparse gradients", "catastrophic forgetting"]

METHOD_SENT = [
    "We propose {method}, which reformulates {TopicNoun} around {mech}.",
    "This paper introduces {method}, combining {mech} with {TopicNoun}.",
    "We present {method}, a {TopicNoun} approach built on {mech}.",
]
MECHS = ["a learned routing objective", "an information bottleneck", "a curriculum over difficulty",
         "closed-form updates", "a contrastive alignment loss", "amortized inference",
         "kernelized feature maps", "a divide-and-conquer decomposition",
         "sparse retrieval priors", "restarts with warm state"]

EXP_SENT = [
    "Across {dataset} and {n} additional benchmarks, {method} improves {metric} by {gain} over the strongest published baseline.",
    "Experiments on {dataset} show {gain} better {metric} while cutting {metric2} by {frac}.",
    "On {dataset}, our approach matches teacher performance and reaches {metric} gains of {gain}, at {frac} of baseline {metric2}.",
]
RESULT_SENT = [
    "Ablations confirm each component contributes: removing {mech} costs {frac2} of the improvement.",
    "We further show the gains persist under distribution shift and reduced supervision.",
    "Qualitative studies reveal interpretable intermediate representations and stable convergence.",
]
CLOSE_SENT = [
    "We release code, model checkpoints, and evaluation harnesses to support reproducibility.",
    "These results suggest {TopicNoun} is a practical primitive for production {task} systems.",
    "We discuss limitations, negative results, and directions for future work.",
]

QUERY_TEMPLATES = [
    "What recent work applies {method} to {task}?",
    "papers on {method} for {task}",
    "{method} improvements for {task}",
    "How do researchers address {issue} in {TopicNoun}?",
    "studies improving {metric} using {method}",
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def gen_docs(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    topic_names = sorted(TOPICS)
    docs = []
    used_titles: set[str] = set()
    i = 0
    while len(docs) < n:
        topic = topic_names[i % len(topic_names)]
        vocab = TOPICS[topic]
        topic_noun = topic
        method = rng.choice(vocab["methods"])
        task = rng.choice(vocab["tasks"])
        dataset = rng.choice(vocab["datasets"])
        metric = rng.choice(vocab["metrics"])
        metric2 = rng.choice([m for m in vocab["metrics"] if m != metric] or ["latency"])
        mech = rng.choice(MECHS)
        issue = rng.choice(ISSUES)
        pattern = rng.choice(TITLE_PATTERNS)
        title = (pattern.format(
            Method=_cap(method), Adj=rng.choice(ADJS), AdjCap=_cap(rng.choice(ADJS)),
            Adjity=rng.choice(ADJITY), Adj2=rng.choice(ADJ2), Scale=rng.choice(SCALES),
            TopicNoun=topic_noun, Task=task))
        if title in used_titles:  # keep cycling until unique enough
            i += 1
            continue
        used_titles.add(title)

        gain = f"{rng.choice([3, 4, 5, 6, 8, 11, 12, 15, 18, 21, 26])}%"
        frac = rng.choice(["one third", "half", "a quarter", "40%"])
        frac2 = rng.choice(["one third", "over half", "most"])
        n_extra = rng.choice([2, 3, 4, 5])
        abstract = " ".join([
            rng.choice(PROBLEM).format(
                TopicNoun=topic_noun, task=task, issue=issue, scale=rng.choice(SCALES)),
            rng.choice(METHOD_SENT).format(method=method, TopicNoun=topic_noun, mech=mech),
            rng.choice(EXP_SENT).format(
                dataset=dataset, n=n_extra, method=method, metric=metric,
                metric2=metric2, gain=gain, frac=frac),
            rng.choice(RESULT_SENT).format(mech=mech, frac2=frac2),
            rng.choice(CLOSE_SENT).format(TopicNoun=topic_noun, task=task),
        ])
        # arXiv-style id: YYMM.NNNNN (2015+ so 5-digit suffixes are valid)
        yy = rng.randint(15, 26)
        mm = rng.randint(1, 12)
        arxiv_id = f"{yy:02d}{mm:02d}.{rng.randint(10, 99999):05d}"
        if arxiv_id in {d["arxiv_id"] for d in docs}:  # ids must be unique
            continue
        cats = [f"cs.{c}" for c in rng.sample(
            ["LG", "CL", "IR", "CV", "DC", "AI", "NE", "SE", "SI", "SY"],
            rng.randint(1, 3))]
        docs.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": cats,
        })
        i += 1
    return docs


def gen_golden(docs: list[dict], n: int, seed: int = 7) -> list[dict]:
    """Deterministic sample spread across the corpus; queries are templated
    from each doc's own vocabulary (method/task/metric), so the gold doc is
    genuinely the best answer but not trivially identical to the query."""
    rng = random.Random(seed)
    step = max(1, len(docs) // n)
    picked = [docs[j] for j in range(0, len(docs), step)][:n]
    golden = []
    for d in picked:
        title = d["title"]
        # recover which query template fits by scanning the abstract for cues
        # simpler: derive keywords from title after the colon/comma
        keywords = title.split(":")[0].lower()
        q = rng.choice([
            f"What recent work uses {keywords}?",
            f"papers about {keywords}",
            f"{keywords} for {title.split(' for ')[-1].lower() if ' for ' in title else 'modern tasks'}",
            f"How is {keywords} applied in recent research?",
        ])
        golden.append({"query": q, "arxiv_id": d["arxiv_id"], "gold_title": title})
    return golden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--golden", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = gen_docs(args.n)
    corpus_path = out_dir / f"corpus_{args.n // 1000}k.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"corpus: {len(docs)} docs -> {corpus_path}")

    golden = gen_golden(docs, args.golden)
    golden_path = out_dir / f"golden_{args.golden}.jsonl"
    with golden_path.open("w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g) + "\n")
    print(f"golden: {len(golden)} queries -> {golden_path}")


if __name__ == "__main__":
    main()
