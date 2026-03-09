# MediQuery: Embedding Fine-Tuning & RAG Evaluation Plan

## Overview

This document describes how to fine-tune both the `BAAI/bge-base-en-v1.5` bi-encoder
embedding model and the `BAAI/bge-reranker-v2-m3` cross-encoder reranker on
Medicare-specific query-chunk pairs, rebuild the FAISS index with the fine-tuned embeddings,
and evaluate the full RAG pipeline end-to-end.

### Datasets

| Dataset | File | Rows | Purpose |
|---|---|---|---|
| Training pairs | `query_golden_chunk_pairs_full.csv` | 24,032 | Fine-tune the bi-encoder and cross-encoder reranker |
| Evaluation QA | `rag_final_eval_500_qa_pairs_v2_claude.csv` | 500 | End-to-end RAG evaluation |

### Training Data Schema (`query_golden_chunk_pairs_full.csv`)

| Column | Description |
|---|---|
| `query_id` | Unique query identifier (e.g., `100.3_0_q1`) |
| `query` | Natural language question |
| `query_type` | One of: policy, yes_no, summary, ncd_reference, criteria, indications, mixed_policy, service_condition, use_case, non_coverage_reason, retired_status |
| `gold_chunk_id` | ID of the correct chunk (e.g., `100.3_0`) |
| `source_id` | NCD/LCD document ID |
| `chunk_idx` | Chunk index within the document |
| `title` | Document title |
| `type` | NCD (4,667) or LCD (19,365) |
| `gold_chunk_excerpt` | The text of the gold-standard chunk |
| `reference_answer` | Ground-truth answer |

### Evaluation Data Schema (`rag_final_eval_500_qa_pairs_v2_claude.csv`)

| Column | Description |
|---|---|
| `qa_id` | Unique QA identifier |
| `question` | Natural language question |
| `question_type` | indications, mixed_policy, criteria, non_coverage_reason, retired_status, policy |
| `answer` | Ground-truth answer text |
| `source_title` | Title of source document |
| `doc_type` | LCD (201), NCD (149), Benefit Policy (51), Claims Processing (99) |
| `coverage_status` | informational (390), mixed (24), covered (61), non-covered (20), retired (5) |
| `chunk_id` | Gold chunk ID for retrieval evaluation |
| `states` | Applicable states |

---

## Phase 1: Prepare Training Data for Fine-Tuning

### Step 1.1 — Load and inspect the 24K pairs

```python
import pandas as pd

train_df = pd.read_csv("query_golden_chunk_pairs_full.csv")
print(f"Total training pairs: {len(train_df):,}")
print(train_df["query_type"].value_counts())
print(train_df["type"].value_counts())
```

### Step 1.2 — Build `(query, positive_passage)` pairs

The fine-tuning objective is **contrastive learning**: teach the model that each `query`
should be close to its `gold_chunk_excerpt` in embedding space. Each row in the CSV
already provides a natural (query, positive) pair.

```python
# Each row becomes one training example
train_pairs = []
for _, row in train_df.iterrows():
    train_pairs.append({
        "query": row["query"],
        "positive": row["gold_chunk_excerpt"]
    })
```

### Step 1.3 — Construct hard negatives via the pre-trained model

Hard negatives dramatically improve contrastive fine-tuning. For each query, retrieve the
top-k chunks from the **existing** FAISS index (before fine-tuning), then select the
highest-scoring chunk that is NOT the gold chunk as the hard negative.

```python
# For each query, retrieve top-20 from pre-trained FAISS index
# Select the highest-ranked chunk whose source_id != gold source_id
for pair in train_pairs:
    results = retrieve_chunks(pair["query"], top_k=20)
    for r in results:
        if r["source_id"] != pair["gold_source_id"]:
            pair["negative"] = r["text"]
            break
```

This produces triplets: `(query, positive_passage, hard_negative)`.

### Step 1.4 — Train/validation split

```python
from sklearn.model_selection import train_test_split

train_set, val_set = train_test_split(
    train_pairs,
    test_size=0.05,      # ~1,200 for validation
    random_state=42,
    stratify=train_df["query_type"]
)
print(f"Train: {len(train_set):,}  |  Val: {len(val_set):,}")
```

### Step 1.5 — Format as `sentence-transformers` InputExample objects

```python
from sentence_transformers import InputExample

train_examples = [
    InputExample(texts=[p["query"], p["positive"], p["negative"]])
    for p in train_set
]
```

---

## Phase 2: Fine-Tune the Embedding Model

### Step 2.1 — Load the base model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
```

### Step 2.2 — Configure the loss function

Use **MultipleNegativesRankingLoss (MNRL)** — the standard loss for contrastive
bi-encoder fine-tuning. When given `(query, positive, hard_negative)` triplets, MNRL
treats other in-batch positives as additional negatives, maximizing training signal.

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss

train_loss = MultipleNegativesRankingLoss(model)
```

### Step 2.3 — Create the DataLoader

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32       # Adjust based on GPU memory (T4: 32, A100: 64-128)
)
```

### Step 2.4 — Set up the evaluation callback

Use the validation set to monitor Recall@5 during training and prevent overfitting.

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Build the IR evaluation dictionaries
queries = {str(i): p["query"] for i, p in enumerate(val_set)}
corpus  = {str(i): p["positive"] for i, p in enumerate(val_set)}
relevant_docs = {str(i): {str(i)} for i in range(len(val_set))}

evaluator = InformationRetrievalEvaluator(
    queries, corpus, relevant_docs,
    name="mediquery-val",
    show_progress_bar=True
)
```

### Step 2.5 — Train

```python
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,                          # 2-4 epochs is typical
    evaluation_steps=500,              # Evaluate every 500 steps
    warmup_steps=200,                  # ~1% of total steps
    output_path="bge-base-mediquery-finetuned",
    save_best_model=True,
    optimizer_params={"lr": 2e-5},     # Conservative LR for fine-tuning
    use_amp=True                       # Mixed precision on GPU
)
```

**Hyperparameter guidance:**

| Parameter | Recommended | Notes |
|---|---|---|
| Epochs | 2–4 | More risks overfitting on 24K pairs |
| Learning rate | 1e-5 to 3e-5 | Lower is safer for pre-trained models |
| Batch size | 32–64 | Larger = more in-batch negatives = better MNRL |
| Warmup steps | 100–300 | Prevents early destabilization |
| Loss | MNRL | Standard for bi-encoder contrastive tuning |

### Step 2.6 — Save and verify

```python
# Model is auto-saved to output_path by save_best_model=True
finetuned_model = SentenceTransformer("bge-base-mediquery-finetuned")
print(f"Embedding dim: {finetuned_model.get_sentence_embedding_dimension()}")
# Should still be 768
```

---

## Phase 3: Rebuild FAISS Index with Fine-Tuned Embeddings

### Step 3.1 — Re-encode all 8,563 chunks

```python
import json, numpy as np, faiss

with open("all_chunks.json") as f:
    all_chunks = json.load(f)

texts = [chunk["text"] for chunk in all_chunks]

finetuned_model = SentenceTransformer("bge-base-mediquery-finetuned", device="cuda")

new_embeddings = finetuned_model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
)
# Verify shape: (8563, 768) and L2 norms ≈ 1.0
```

### Step 3.2 — Build new FAISS index

```python
dimension = new_embeddings.shape[1]
new_index = faiss.IndexFlatIP(dimension)
new_index.add(new_embeddings.astype("float32"))

faiss.write_index(new_index, "faiss_index/medicare_finetuned.index")
np.save("faiss_index/embeddings_finetuned.npy", new_embeddings)
```

### Step 3.3 — Sanity-check retrieval with the same 3 test queries

Run the same queries from the original notebook to compare before/after:

1. *"Does Medicare cover acupuncture for chronic lower back pain?"*
2. *"Is home oxygen therapy covered for patients in Texas?"*
3. *"What are the requirements for skilled nursing facility coverage?"*

Compare top-5 results and cosine scores between the original and fine-tuned models.

---

## Phase 4: Fine-Tune the Cross-Encoder Reranker

The cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is the final precision gate that
decides which 5 chunks reach the LLM. Fine-tuning it on Medicare-specific relevance
judgments teaches the model to better distinguish between truly relevant policy text and
semantically similar but irrelevant passages.

### Step 4.1 — Prepare reranker training data

Cross-encoders are trained on `(query, passage, label)` pairs where `label = 1` means
relevant and `label = 0` means irrelevant. We reuse the same 24K dataset and the hard
negatives mined in Phase 1.

```python
from sentence_transformers import InputExample

reranker_train_examples = []

for pair in train_set:
    # Positive pair: query + gold chunk → label 1
    reranker_train_examples.append(
        InputExample(texts=[pair["query"], pair["positive"]], label=1.0)
    )
    # Negative pair: query + hard negative → label 0
    reranker_train_examples.append(
        InputExample(texts=[pair["query"], pair["negative"]], label=0.0)
    )

print(f"Reranker training examples: {len(reranker_train_examples):,}")
# Expected: ~45,600 (2 × ~22,800 train pairs)
```

### Step 4.2 — Load the base cross-encoder

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
```

### Step 4.3 — Create DataLoader and configure training

```python
from torch.utils.data import DataLoader

reranker_dataloader = DataLoader(
    reranker_train_examples,
    shuffle=True,
    batch_size=16       # Cross-encoders use more memory; 16 is safe on T4
)
```

### Step 4.4 — Set up evaluation callback

Use the validation set to monitor reranker accuracy during training.

```python
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# Build eval data: {query_id: {"query": str, "positive": set, "negative": set}}
reranker_eval_samples = []
for i, pair in enumerate(val_set):
    reranker_eval_samples.append({
        "query": pair["query"],
        "positive": [pair["positive"]],
        "negative": [pair["negative"]]
    })

reranker_evaluator = CERerankingEvaluator(
    reranker_eval_samples,
    name="mediquery-reranker-val"
)
```

### Step 4.5 — Train the reranker

```python
reranker.fit(
    train_dataloader=reranker_dataloader,
    evaluator=reranker_evaluator,
    epochs=2,                          # 1-3 epochs; cross-encoders overfit faster
    evaluation_steps=500,
    warmup_steps=100,
    output_path="bge-reranker-mediquery-finetuned",
    save_best_model=True,
    optimizer_params={"lr": 1e-5},     # Lower LR than bi-encoder
    use_amp=True
)
```

**Hyperparameter guidance:**

| Parameter | Recommended | Notes |
|---|---|---|
| Epochs | 1–3 | Cross-encoders overfit faster than bi-encoders |
| Learning rate | 5e-6 to 2e-5 | More conservative than bi-encoder tuning |
| Batch size | 16–32 | Cross-encoders are memory-heavy (both texts processed jointly) |
| Warmup steps | 50–200 | Short warmup is sufficient |
| Max length | 512 | Matches the base model's training config |

### Step 4.6 — Save and verify

```python
finetuned_reranker = CrossEncoder("bge-reranker-mediquery-finetuned")

# Quick sanity check: score a known relevant and irrelevant pair
relevant_score = finetuned_reranker.predict([
    ("Does Medicare cover acupuncture?", "<gold acupuncture chunk text>")
])
irrelevant_score = finetuned_reranker.predict([
    ("Does Medicare cover acupuncture?", "<unrelated wound care chunk text>")
])
print(f"Relevant: {relevant_score[0]:.4f}  |  Irrelevant: {irrelevant_score[0]:.4f}")
# Relevant score should be >> irrelevant score
```

---

## Phase 5: Retrieval Evaluation (Before vs. After Fine-Tuning)

### Step 5.1 — Build the gold-standard retrieval mapping from the eval set

```python
eval_df = pd.read_csv("rag_final_eval_500_qa_pairs_v2_claude.csv")

# Map each question to its gold chunk_id
gold_map = {}
for _, row in eval_df.iterrows():
    gold_map[row["qa_id"]] = {
        "question": row["question"],
        "gold_chunk_id": row["chunk_id"],
        "doc_type": row["doc_type"],
        "states": row["states"]
    }
```

### Step 5.2 — Run retrieval with BOTH models on all 500 queries

```python
def evaluate_retrieval(model, index, all_chunks, eval_df, top_k=5):
    """Compute Recall@k and MRR over the eval set."""
    hits = 0
    reciprocal_ranks = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        gold_id = row["chunk_id"]

        results = retrieve_chunks_with_model(query, model, index, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]

        if gold_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(gold_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = hits / len(eval_df)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall_at_k, mrr
```

### Step 5.3 — Compare results

| Metric | Pre-Trained BGE | Fine-Tuned BGE | Delta |
|---|---|---|---|
| Recall@5 | [run] | [run] | |
| Recall@10 | [run] | [run] | |
| Recall@20 | [run] | [run] | |
| MRR | [run] | [run] | |

Also break down by `doc_type` (NCD, LCD, Benefit Policy, Claims Processing) and
`coverage_status` to see where fine-tuning helped most.

---

## Phase 6: End-to-End RAG Evaluation on 500 QA Pairs

### Step 6.1 — Define the three system configurations

| Config | Retrieval | Reranker | Generator |
|---|---|---|---|
| **Baseline LLM** | None | None | Mistral-7B-Instruct (no context) |
| **RAG (fine-tuned)** | Fine-tuned BGE → FAISS → top-20 → Fine-tuned bge-reranker → top-5 | Yes (fine-tuned) | Mistral-7B-Instruct |
| **RAG + Query Rewrite** | Same as above, with query rewriting step | Yes (fine-tuned) | Mistral-7B-Instruct |

### Step 6.2 — Run all 500 queries through each configuration

```python
results_all = []

for _, row in eval_df.iterrows():
    question = row["question"]
    gold_answer = row["answer"]
    gold_chunk = row["chunk_id"]

    # Config 1: Baseline (no retrieval)
    baseline_answer = generate_baseline(question)

    # Config 2: RAG with fine-tuned bi-encoder + fine-tuned reranker
    rag_chunks = retrieve_and_rerank(
        question, embed_model=finetuned_model, reranker=finetuned_reranker
    )
    rag_answer = generate_with_context(question, rag_chunks)

    # Config 3: RAG + query rewriting (same fine-tuned models)
    rewritten_q = rewrite_query(question)
    rqr_chunks = retrieve_and_rerank(
        rewritten_q, embed_model=finetuned_model, reranker=finetuned_reranker
    )
    rqr_answer = generate_with_context(question, rqr_chunks)

    results_all.append({
        "qa_id": row["qa_id"],
        "question": question,
        "gold_answer": gold_answer,
        "gold_chunk_id": gold_chunk,
        "baseline_answer": baseline_answer,
        "rag_answer": rag_answer,
        "rag_chunks": rag_chunks,
        "rqr_answer": rqr_answer,
        "rqr_chunks": rqr_chunks
    })
```

### Step 6.3 — Compute evaluation metrics

#### Retrieval Metrics (RAG and RAG+Rewrite only)
- **Recall@5**: % of questions where gold chunk appears in top 5
- **MRR**: Mean reciprocal rank of the gold chunk

#### Generation Metrics (all 3 configs)
- **Answer Correctness**: Binary human judgment or LLM-as-judge (does the answer match the gold answer?)
- **Faithfulness**: Are all claims in the generated answer supported by the retrieved chunks?
- **Citation Accuracy**: Does the cited NCD/LCD ID match the actual source document?
- **Hallucination Rate**: % of answers with at least one unsupported factual claim

```python
# LLM-as-judge for faithfulness scoring
def judge_faithfulness(question, generated_answer, retrieved_chunks):
    """Use an LLM to score whether the answer is faithful to the evidence."""
    prompt = f"""You are an expert evaluator. Given the question, the generated answer,
and the retrieved evidence chunks, score faithfulness on a scale of 0-1.

Question: {question}
Generated Answer: {generated_answer}
Evidence: {retrieved_chunks}

Score (0=unfaithful, 1=fully faithful):"""
    # Call judge LLM
    ...

# Citation accuracy check
def check_citation(generated_answer, gold_source_id):
    """Check if the answer cites the correct source document."""
    cited_ids = extract_cited_ids(generated_answer)  # regex for NCD/LCD IDs
    return gold_source_id in cited_ids
```

### Step 6.4 — Fill in the results table

| System | Recall@5 | MRR | Correctness | Faithfulness | Citation Acc. | Halluc. Rate |
|---|---|---|---|---|---|---|
| Baseline LLM | — | — | ___ | ___ | ___ | ___ |
| RAG (fine-tuned) | ___ | ___ | ___ | ___ | ___ | ___ |
| RAG + Query Rewrite | ___ | ___ | ___ | ___ | ___ | ___ |

### Step 6.5 — Breakdown analysis

Compute per-category metrics to identify strengths and weaknesses:

**By question type:**
| Question Type | Count | RAG Recall@5 | RAG Correctness |
|---|---|---|---|
| indications | 277 | | |
| criteria | 99 | | |
| policy | 75 | | |
| mixed_policy | 24 | | |
| non_coverage_reason | 20 | | |
| retired_status | 5 | | |

**By document type:**
| Doc Type | Count | RAG Recall@5 | RAG Correctness |
|---|---|---|---|
| LCD | 201 | | |
| NCD | 149 | | |
| Claims Processing | 99 | | |
| Benefit Policy | 51 | | |

---

## Phase 7: Ablation — Pre-Trained vs. Fine-Tuned Component Comparison

Run the same 500-query evaluation with different combinations of pre-trained and
fine-tuned components to isolate the impact of each fine-tuning step.

### Step 7.1 — Define ablation configurations

| Config | Bi-Encoder | Reranker | Purpose |
|---|---|---|---|
| A | Pre-trained BGE | Pre-trained reranker | Baseline (no fine-tuning) |
| B | **Fine-tuned** BGE | Pre-trained reranker | Isolate bi-encoder improvement |
| C | Pre-trained BGE | **Fine-tuned** reranker | Isolate reranker improvement |
| D | **Fine-tuned** BGE | **Fine-tuned** reranker | Full fine-tuned pipeline |

### Step 7.2 — Run all 4 configs on the 500 eval queries

```python
configs = {
    "A_pretrained_both":   (pretrained_model, pretrained_reranker),
    "B_finetuned_bienc":   (finetuned_model,  pretrained_reranker),
    "C_finetuned_rerank":  (pretrained_model,  finetuned_reranker),
    "D_finetuned_both":    (finetuned_model,  finetuned_reranker),
}

ablation_results = {}
for name, (embed_model, reranker) in configs.items():
    recall, mrr = evaluate_retrieval_with_reranking(
        embed_model, reranker, eval_df, top_k_retrieve=20, top_k_rerank=5
    )
    ablation_results[name] = {"recall@5": recall, "mrr": mrr}
```

### Step 7.3 — Compare results

| Metric | A: Both Pre-trained | B: FT Bi-Encoder Only | C: FT Reranker Only | D: Both Fine-Tuned |
|---|---|---|---|---|
| Recall@5 | | | | |
| Recall@10 | | | | |
| MRR | | | | |
| Faithfulness | | | | |
| Citation Accuracy | | | | |
| Hallucination Rate | | | | |

This table reveals whether gains come primarily from the bi-encoder, the reranker, or
their combination.

---

## Summary of Notebook Cells to Add/Modify

The following cells should be added to a **new notebook** (or appended to the existing
`2-3_Embeddings_FAISS_MediQuery_till_reranking.ipynb`):

| Cell | Phase | Description |
|---|---|---|
| 1 | Setup | Install deps: `sentence-transformers`, `faiss-cpu`, `pandas`, `scikit-learn` |
| 2 | Phase 1 | Load `query_golden_chunk_pairs_full.csv`, inspect statistics |
| 3 | Phase 1 | Build (query, positive) pairs from the dataset |
| 4 | Phase 1 | Mine hard negatives using the pre-trained FAISS index |
| 5 | Phase 1 | Train/val split (95/5), create InputExample objects |
| 6 | Phase 2 | Load base bi-encoder, configure MNRL loss, create DataLoader |
| 7 | Phase 2 | Set up InformationRetrievalEvaluator on val set |
| 8 | Phase 2 | Run `model.fit()` — fine-tune bi-encoder for 2-4 epochs |
| 9 | Phase 2 | Save fine-tuned bi-encoder, verify embedding dimension |
| 10 | Phase 3 | Re-encode all 8,563 chunks with the fine-tuned bi-encoder |
| 11 | Phase 3 | Build and save new FAISS index |
| 12 | Phase 3 | Sanity-check: run 3 sample queries, compare with baseline |
| 13 | Phase 4 | Prepare reranker training data (positive + negative pairs with labels) |
| 14 | Phase 4 | Load base cross-encoder, configure DataLoader |
| 15 | Phase 4 | Set up CERerankingEvaluator on val set |
| 16 | Phase 4 | Run `reranker.fit()` — fine-tune cross-encoder for 1-3 epochs |
| 17 | Phase 4 | Save fine-tuned reranker, sanity-check scores |
| 18 | Phase 5 | Load `rag_final_eval_500_qa_pairs_v2_claude.csv` |
| 19 | Phase 5 | Compute Recall@5, Recall@10, MRR for both models on 500 queries |
| 20 | Phase 5 | Print comparison table, per-category breakdown |
| 21 | Phase 6 | Run 500 queries through Baseline, RAG, RAG+Rewrite configs |
| 22 | Phase 6 | Compute generation metrics (correctness, faithfulness, citation, hallucination) |
| 23 | Phase 6 | Print final results table matching the `main.tex` format |
| 24 | Phase 7 | Ablation: 4-way comparison (pre-trained vs fine-tuned, bi-encoder × reranker) |

---

## Expected Compute Requirements

| Task | Time (T4 GPU) | Time (A100 GPU) |
|---|---|---|
| Hard negative mining (24K queries × FAISS search) | ~15 min | ~5 min |
| Bi-encoder fine-tuning (3 epochs, batch=32, 24K examples) | ~45 min | ~15 min |
| Re-encoding 8,563 chunks | ~2 min | ~30 sec |
| Reranker fine-tuning (2 epochs, batch=16, ~46K pairs) | ~60 min | ~20 min |
| 500-query retrieval eval (per config) | ~3 min | ~1 min |
| 500-query ablation (4 configs) | ~12 min | ~4 min |
| 500 LLM generations (Mistral-7B) | ~60 min | ~20 min |

**Total estimated time: ~3–4 hours on Colab T4, ~1.5 hours on A100.**
