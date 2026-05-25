# MediQuery - Medicare Policy RAG System

> A Retrieval-Augmented Generation system for answering natural language questions about Medicare coverage, grounded in official CMS policy documents.



---

## The Problem

Navigating Medicare coverage requires searching through thousands of fragmented federal and state-level policy documents written in dense regulatory language. A provider asking *"Does Medicare cover physical therapy after knee replacement in Texas?"* must manually cross-reference national and local policies - a process that is time-consuming, error-prone, and inaccessible to non-specialists.

General-purpose LLMs cannot reliably answer these questions either. They hallucinate coverage determinations that may not exist, with no citations to verify.

**MediQuery solves this by grounding every answer in retrieved, official CMS documents - with explicit citations.**

---

## System Architecture

```
+----------------------------------------------------------+
|                   Streamlit / React Frontend             |
+----------------------------------------------------------+
                             |
                             v
+----------------------------------------------------------+
|               FastAPI Backend                            |
|                                                          |
|   Query --> [Optional] Query Rewriter (LLM)              |
|                             |                            |
|                             v                            |
|         +-----------------------------------+            |
|         |     Stage 1: Dense Retrieval      |            |
|         |                                   |            |
|         |  Embed query (bge-base-en-v1.5)   |            |
|         |  --> FAISS IndexFlatIP search      |            |
|         |  --> Top 20 candidate chunks       |            |
|         |  --> State-aware filtering (LCD)   |            |
|         +-----------------------------------+            |
|                             |                            |
|                             v                            |
|         +-----------------------------------+            |
|         |   Stage 2: Cross-Encoder Rerank   |            |
|         |                                   |            |
|         |  Score (query, chunk) pairs        |            |
|         |  bge-reranker-v2-m3               |            |
|         |  --> Top 5 evidence chunks         |            |
|         +-----------------------------------+            |
|                             |                            |
|                             v                            |
|         +-----------------------------------+            |
|         |     LLM Grounded Generation       |            |
|         |                                   |            |
|         |  Mistral-7B-Instruct              |            |
|         |  Structured JSON output            |            |
|         |  { answer, citations }             |            |
|         |  Fallback: "Insufficient evidence" |            |
|         +-----------------------------------+            |
+----------------------------------------------------------+
                             |
                             v
+----------------------------------------------------------+
|                Knowledge Base (8,563 chunks)             |
|                                                          |
|   National Coverage Determinations (NCDs)  343 docs      |
|   Local Coverage Determinations (LCDs)     819 docs      |
|   Medicare Benefit Policy Manual            17 chapters  |
|   Medicare Claims Processing Manual        39 chapters   |
|                                                          |
|   Each chunk tagged with: source_id, title,              |
|   document type, states, contractor, chunk_idx           |
+----------------------------------------------------------+
```

---

## Results

| Metric | Pre-trained | Fine-tuned | Delta |
|---|---|---|---|
| Recall@5 (FAISS) | 0.646 | 0.785 | +0.139 |
| Recall@5 (Reranked) | 0.672 | **0.810** | +0.138 |
| MRR (Reranked) | 0.475 | **0.630** | +0.155 |
| NDCG@5 (Reranked) | 0.524 | **0.665** | +0.141 |

Fine-tuning on 56,288 Medicare-specific query-chunk pairs produced consistent double-digit improvements across every retrieval metric. The correct evidence chunk appears in the top-5 context for **over 81% of test queries**.

---

## Pipeline Details

### 1. Knowledge Base Construction

Corpus sourced entirely from the [CMS Medicare Coverage Database](https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx) and CMS Internet-Only Manuals:

- **343 NCDs** - federal-level coverage rulings, parsed from structured CSV bulk export
- **819 LCDs** - regional policies by Medicare Administrative Contractor (MAC), parsed from Excel with state metadata extraction
- **17 chapters** - Medicare Benefit Policy Manual (PDF, via PyMuPDF)
- **39 chapters** - Medicare Claims Processing Manual (PDF, via PyMuPDF)

Each document chunked at **400 words with 50-word overlap**. Every chunk is prefixed with a structured metadata header:

```
[TYPE: LCD | LCD_ID: 35125 | TITLE: Wound Care | STATES: TX, AR, CO, LA | CONTRACTOR: Novitas]
```

This allows the LLM to generate citations directly from the chunk content without any additional lookup.

**Final corpus: 8,563 chunks from 1,218 documents.**

### 2. Embeddings and Vector Index

- Embedding model: `BAAI/bge-base-en-v1.5` (768-dim, L2-normalized)
- Index: `FAISS IndexFlatIP` - exact cosine search, no approximation needed at this scale
- Metadata-tagged chunk text is embedded (not raw text), so a query like "wound care Texas" matches on both content and state tags
- Index size: 25.1 MB

### 3. Fine-Tuning

Both the bi-encoder and cross-encoder were fine-tuned on Medicare-specific training data:

**Bi-encoder fine-tuning:**
- Loss: `MultipleNegativesRankingLoss` (temperature = 0.02, scale = 50)
- Training set: 48,126 triplets (query, positive chunk, hard negative)
- Hard negatives mined from pre-trained FAISS index - semantically close but wrong source document
- Best checkpoint selected by `cosine_ndcg@10` on 2,533 validation queries

**Cross-encoder fine-tuning:**
- Loss: `BinaryCrossEntropyLoss`
- Hard negatives re-mined from the fine-tuned bi-encoder index (up to 7 per query)
- Training set: 307,877 pairwise examples
- Best checkpoint selected by `NDCG@5`

### 4. Retrieval Pipeline

```
Query
  --> Embed with fine-tuned bge-base-en-v1.5
  --> FAISS search: top 20 candidates
  --> State-aware filtering (if state mentioned in query, prioritize matching LCD chunks)
  --> Cross-encoder rerank: score all (query, chunk) pairs
  --> Top 5 evidence chunks passed to generator
```

### 5. Grounded Generation

- Generator: `Mistral-7B-Instruct` (via REST API)
- Prompt enforces: use only retrieved evidence, no invented facts, structured JSON output
- Output schema: `{ "answer": "...", "citations": [...] }`
- Explicit fallback: `"Insufficient evidence in retrieved documents."` - model returns this rather than guessing when evidence is weak

### 6. Evaluation

Three system configurations evaluated:

| Configuration | Description |
|---|---|
| Baseline LLM | Ungrounded LLM, no retrieval |
| RAG | Dense retrieval + reranking + grounded generation |
| RAG + Query Rewriting | Query rewritten before retrieval |

RAG substantially reduces hallucination and improves citation accuracy compared to the ungrounded baseline. Evaluation dataset: 500 hand-curated question-answer pairs, stratified by document type and coverage status.

---

## Tech Stack

| Component | Tool |
|---|---|
| Embedding | `sentence-transformers`, `BAAI/bge-base-en-v1.5` |
| Vector Search | `FAISS` (IndexFlatIP) |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Fine-tuning | `SentenceTransformerTrainer`, `MultipleNegativesRankingLoss` |
| Generation | `Mistral-7B-Instruct` |
| Backend | `FastAPI` |
| Frontend | `React`, `Streamlit` |
| Data Processing | `BeautifulSoup`, `openpyxl`, `PyMuPDF` |
| Compute | Google Colab Pro (GPU) |

---

## Example Queries

```
"Does Medicare cover acupuncture for chronic lower back pain?"
--> NCD 30.1: Yes, up to 12 sessions within 90 days for pain lasting 12+ weeks,
    with 8 additional sessions if measurable improvement shown.

"Is home oxygen therapy covered for patients in Texas?"
--> NCD 240.2 + LCD 33797 (Novitas, TX): Coverage requires physician order,
    documented hypoxemia, and qualifying diagnosis.

"Does Medicare cover continuous glucose monitors?"
--> NCD 160.18: Yes, for insulin-treated diabetes with documented medical necessity.
```

---

## Team

Built by Mohar Chaudhuri, Shruthi Chembu Kuppuswamy, Sanchal Nachappa, Janani Vakkanti, and Ryota Yokoyama as part of GEN AI 285N / MSBA 285 at UT Austin McCombs School of Business (March 2026).

---

## Data Source

All policy documents sourced from the [Centers for Medicare & Medicaid Services (CMS)](https://www.cms.gov). This system is a decision-support tool and does not constitute medical or legal advice.
