"""Split 4_Fine_Tuning_and_Evaluation.ipynb into 4A and 4B notebooks."""
import json, copy

SRC = "C:/Users/ry092/Desktop/MIS 285N/Final Project/MediQuery-RAG-/4_Fine_Tuning_and_Evaluation.ipynb"
OUT_A = "C:/Users/ry092/Desktop/MIS 285N/Final Project/MediQuery-RAG-/4A_Fine_Tuning.ipynb"
OUT_B = "C:/Users/ry092/Desktop/MIS 285N/Final Project/MediQuery-RAG-/4B_RAG_Evaluation.ipynb"

with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK A: cells 0-62 (fine-tuning + retrieval eval)
# ═══════════════════════════════════════════════════════════════════════════════
nb_a = copy.deepcopy(nb)
nb_a["cells"] = [copy.deepcopy(c) for c in cells[:63]]

for c in nb_a["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

# Update title
nb_a["cells"][0]["source"] = [
    "# Step 4A: Fine-Tuning Bi-Encoder & Reranker\n",
    "\n",
    "This notebook implements the fine-tuning pipeline for MediQuery:\n",
    "\n",
    "1. **Phase 1** \u2014 Prepare training data (24K query\u2013golden-chunk pairs + hard negatives)\n",
    "2. **Phase 2** \u2014 Fine-tune the bi-encoder (`BAAI/bge-base-en-v1.5`)\n",
    "3. **Phase 3** \u2014 Rebuild the FAISS index with fine-tuned embeddings\n",
    "4. **Phase 4** \u2014 Fine-tune the cross-encoder reranker (`BAAI/bge-reranker-v2-m3`)\n",
    "5. **Phase 5** \u2014 Retrieval evaluation (pre-trained vs. fine-tuned)\n",
    "\n",
    "### Prerequisites\n",
    "\n",
    "This notebook assumes that the **Step 2\u20133 notebook** has already been run, producing:\n",
    "\n",
    "| Artifact | Path |\n",
    "|---|---|\n",
    "| Chunk corpus | `{DATA_DIR}/all_chunks.json` |\n",
    "| Pre-trained FAISS index | `{DATA_DIR}/faiss_index/medicare.index` |\n",
    "| Chunk metadata | `{DATA_DIR}/faiss_index/chunk_metadata.json` |\n",
    "| Document IDs | `{DATA_DIR}/faiss_index/docids.txt` |\n",
    "| Embeddings | `{DATA_DIR}/faiss_index/embeddings.npy` |\n",
    "\n",
    "### Outputs\n",
    "\n",
    "After running, the following are saved for **Step 4B** (End-to-End RAG Evaluation):\n",
    "\n",
    "| Artifact | Path |\n",
    "|---|---|\n",
    "| Fine-tuned bi-encoder | `{DATA_DIR}/bge-base-mediquery-finetuned/` |\n",
    "| Fine-tuned FAISS index | `{DATA_DIR}/faiss_index/medicare_finetuned.index` |\n",
    "| Fine-tuned embeddings | `{DATA_DIR}/faiss_index/embeddings_finetuned.npy` |\n",
    "| Fine-tuned reranker | `{DATA_DIR}/bge-reranker-mediquery-finetuned/` |",
]

with open(OUT_A, "w", encoding="utf-8") as f:
    json.dump(nb_a, f, indent=1, ensure_ascii=False)

print(f"Notebook A: {len(nb_a['cells'])} cells -> {OUT_A}")


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK B: new setup + cells 63-75 (E2E RAG eval + ablation)
# ═══════════════════════════════════════════════════════════════════════════════
nb_b = copy.deepcopy(nb)
nb_b["cells"] = []


def md(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines}


def code(source_lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source_lines,
    }


# ── Title ────────────────────────────────────────────────────────────────────
nb_b["cells"].append(md([
    "# Step 4B: End-to-End RAG Evaluation & Ablation\n",
    "\n",
    "This notebook runs the **end-to-end RAG evaluation** and **ablation study** for MediQuery.\n",
    "\n",
    "1. **Phase 6** \u2014 End-to-end RAG evaluation on 500 QA pairs (Baseline LLM, RAG, RAG + Query Rewrite)\n",
    "2. **Phase 7** \u2014 Ablation study (isolating bi-encoder vs. reranker fine-tuning gains)\n",
    "\n",
    "### Prerequisites\n",
    "\n",
    "This notebook requires artifacts from **Step 4A** (`4A_Fine_Tuning.ipynb`):\n",
    "\n",
    "| Artifact | Path |\n",
    "|---|---|\n",
    "| Fine-tuned bi-encoder | `{DATA_DIR}/bge-base-mediquery-finetuned/` |\n",
    "| Fine-tuned FAISS index | `{DATA_DIR}/faiss_index/medicare_finetuned.index` |\n",
    "| Fine-tuned reranker | `{DATA_DIR}/bge-reranker-mediquery-finetuned/` |\n",
    "| Chunk corpus | `{DATA_DIR}/all_chunks.json` |\n",
    "| Pre-trained FAISS index | `{DATA_DIR}/faiss_index/medicare.index` |\n",
    "| Chunk metadata | `{DATA_DIR}/faiss_index/chunk_metadata.json` |\n",
    "| Eval QA pairs | `{DATA_DIR}/datasets/rag_final_eval_500_qa_pairs_v2_claude.csv` |",
]))

# ── Install ──────────────────────────────────────────────────────────────────
nb_b["cells"].append(md(["## Install Dependencies"]))
nb_b["cells"].append(code(["!pip install -q sentence-transformers faiss-cpu pandas scikit-learn"]))

# ── Mount Drive ──────────────────────────────────────────────────────────────
nb_b["cells"].append(md(["## Mount Google Drive & Path Configuration"]))
nb_b["cells"].append(code([
    "from google.colab import drive\n",
    "drive.mount('/content/drive', force_remount=True)",
]))

# ── Paths + verification ────────────────────────────────────────────────────
nb_b["cells"].append(code([
    "import os\n",
    "\n",
    "# --- PATH CONFIGURATION (matches Step 4A notebook) ---\n",
    "DATA_DIR    = '/content/drive/MyDrive/Embeddings'\n",
    "CHUNKS_FILE = f'{DATA_DIR}/all_chunks.json'\n",
    "OUTPUT_DIR  = f'{DATA_DIR}/faiss_index'\n",
    "INDEX_FILE  = f'{OUTPUT_DIR}/medicare.index'\n",
    "META_FILE   = f'{OUTPUT_DIR}/chunk_metadata.json'\n",
    "\n",
    "# --- FINE-TUNED MODEL PATHS (produced by Step 4A) ---\n",
    "FT_BIENCODER_DIR = f'{DATA_DIR}/bge-base-mediquery-finetuned'\n",
    "FT_RERANKER_DIR  = f'{DATA_DIR}/bge-reranker-mediquery-finetuned'\n",
    "FT_INDEX_FILE    = f'{OUTPUT_DIR}/medicare_finetuned.index'\n",
    "\n",
    "# --- EVAL DATA ---\n",
    "DATASETS_DIR = f'{DATA_DIR}/datasets'\n",
    "EVAL_CSV     = f'{DATASETS_DIR}/rag_final_eval_500_qa_pairs_v2_claude.csv'\n",
    "\n",
    "# --- Verify all required artifacts exist ---\n",
    "required = {\n",
    "    'Chunk corpus':            CHUNKS_FILE,\n",
    "    'Chunk metadata':          META_FILE,\n",
    "    'Pre-trained FAISS index': INDEX_FILE,\n",
    "    'Fine-tuned bi-encoder':   FT_BIENCODER_DIR,\n",
    "    'Fine-tuned FAISS index':  FT_INDEX_FILE,\n",
    "    'Fine-tuned reranker':     FT_RERANKER_DIR,\n",
    "    'Eval CSV':                EVAL_CSV,\n",
    "}\n",
    "\n",
    "print('Artifact check:')\n",
    "all_ok = True\n",
    "for name, path in required.items():\n",
    "    exists = os.path.exists(path)\n",
    "    status = 'OK' if exists else 'MISSING'\n",
    "    if not exists:\n",
    "        all_ok = False\n",
    "    print(f'  {status:7s} | {name:28s} | {path}')\n",
    "\n",
    "if not all_ok:\n",
    "    print('\\nERROR: Some artifacts are missing. Run Step 4A first.')\n",
    "else:\n",
    "    print('\\nAll artifacts found.')",
]))

# ── Load all models and data ────────────────────────────────────────────────
nb_b["cells"].append(md([
    "## Load All Models and Data\n",
    "\n",
    "Load every artifact needed for evaluation:\n",
    "- Chunk corpus + metadata\n",
    "- Pre-trained bi-encoder + FAISS index (for ablation)\n",
    "- Fine-tuned bi-encoder + FAISS index\n",
    "- Pre-trained reranker (for ablation)\n",
    "- Fine-tuned reranker\n",
    "- 500-question eval set",
]))
nb_b["cells"].append(code([
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import faiss\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "from sentence_transformers import SentenceTransformer, CrossEncoder\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "print(f'Device: {device}')\n",
    "if device == 'cuda':\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
    "\n",
    "# --- Chunk corpus and metadata ---\n",
    "with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:\n",
    "    all_chunks = json.load(f)\n",
    "with open(META_FILE, 'r', encoding='utf-8') as f:\n",
    "    chunk_metadata = json.load(f)\n",
    "print(f'Chunks loaded: {len(all_chunks):,}')\n",
    "\n",
    "# --- Pre-trained models (for ablation comparison) ---\n",
    "pretrained_embed_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)\n",
    "pretrained_index = faiss.read_index(INDEX_FILE)\n",
    "pretrained_reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device, max_length=512)\n",
    "print(f'Pre-trained bi-encoder loaded: BAAI/bge-base-en-v1.5')\n",
    "print(f'Pre-trained FAISS index: {pretrained_index.ntotal:,} vectors')\n",
    "print(f'Pre-trained reranker loaded: BAAI/bge-reranker-v2-m3')\n",
    "\n",
    "# --- Fine-tuned models (from Step 4A) ---\n",
    "finetuned_embed_model = SentenceTransformer(FT_BIENCODER_DIR, device=device)\n",
    "finetuned_index = faiss.read_index(FT_INDEX_FILE)\n",
    "finetuned_reranker = CrossEncoder(FT_RERANKER_DIR, device=device, max_length=512)\n",
    "print(f'Fine-tuned bi-encoder loaded from: {FT_BIENCODER_DIR}')\n",
    "print(f'Fine-tuned FAISS index: {finetuned_index.ntotal:,} vectors')\n",
    "print(f'Fine-tuned reranker loaded from: {FT_RERANKER_DIR}')\n",
    "\n",
    "# --- Eval dataset ---\n",
    "eval_df = pd.read_csv(EVAL_CSV)\n",
    "print(f'\\nEval set loaded: {len(eval_df)} questions')\n",
    "print(eval_df['question_type'].value_counts())",
]))

# ── Helper functions ─────────────────────────────────────────────────────────
nb_b["cells"].append(md(["## Helper Functions"]))
nb_b["cells"].append(code([
    "def retrieve_chunks(query, embed_model, index, all_chunks, chunk_metadata, top_k=20):\n",
    "    \"\"\"Embed a query and retrieve top-k chunks from a FAISS index.\"\"\"\n",
    "    query_vec = embed_model.encode(\n",
    "        query, normalize_embeddings=True, convert_to_numpy=True\n",
    "    ).astype('float32').reshape(1, -1)\n",
    "\n",
    "    scores, indices = index.search(query_vec, top_k)\n",
    "\n",
    "    results = []\n",
    "    for rank, idx in enumerate(indices[0]):\n",
    "        if idx == -1:\n",
    "            continue\n",
    "        meta = chunk_metadata[idx]\n",
    "        results.append({\n",
    "            'faiss_score': float(scores[0][rank]),\n",
    "            'text':        all_chunks[idx]['text'],\n",
    "            'title':       meta['title'],\n",
    "            'type':        meta['type'],\n",
    "            'states':      meta.get('states', ['ALL']),\n",
    "            'source_id':   meta['source_id'],\n",
    "            'chunk_idx':   meta['chunk_idx'],\n",
    "            'chunk_id':    f\"{meta['source_id']}_{meta['chunk_idx']}\"\n",
    "        })\n",
    "    return results\n",
    "\n",
    "\n",
    "def filter_by_state(results, state=None):\n",
    "    \"\"\"Filter retrieved chunks to those covering a specific state.\"\"\"\n",
    "    if state is None:\n",
    "        return results\n",
    "    return [r for r in results if 'ALL' in r['states'] or state in r['states']]\n",
    "\n",
    "\n",
    "def rerank_results(query, results, reranker_model, top_n=5):\n",
    "    \"\"\"Rerank retrieved chunks using a cross-encoder and deduplicate by source_id.\"\"\"\n",
    "    if not results:\n",
    "        return []\n",
    "\n",
    "    pairs = [(query, r['text']) for r in results]\n",
    "    scores = reranker_model.predict(pairs)\n",
    "\n",
    "    for i in range(len(results)):\n",
    "        results[i]['rerank_score'] = float(scores[i])\n",
    "\n",
    "    results.sort(key=lambda x: x['rerank_score'], reverse=True)\n",
    "\n",
    "    seen = set()\n",
    "    final = []\n",
    "    for r in results:\n",
    "        if r['source_id'] not in seen:\n",
    "            final.append(r)\n",
    "            seen.add(r['source_id'])\n",
    "        if len(final) == top_n:\n",
    "            break\n",
    "    return final\n",
    "\n",
    "\n",
    "def retrieve_and_rerank(query, embed_model, faiss_index, reranker_model,\n",
    "                        all_chunks, chunk_metadata, state=None,\n",
    "                        top_k_retrieve=20, top_n_rerank=5):\n",
    "    \"\"\"Full retrieval pipeline: embed -> FAISS -> state filter -> rerank.\"\"\"\n",
    "    retrieved = retrieve_chunks(query, embed_model, faiss_index,\n",
    "                                all_chunks, chunk_metadata, top_k=top_k_retrieve)\n",
    "    filtered = filter_by_state(retrieved, state)\n",
    "    reranked = rerank_results(query, filtered, reranker_model, top_n=top_n_rerank)\n",
    "    return reranked\n",
    "\n",
    "\n",
    "def evaluate_retrieval(embed_model, faiss_index, reranker_model,\n",
    "                       all_chunks, chunk_metadata, eval_df,\n",
    "                       top_k_retrieve=20, top_n_rerank=5):\n",
    "    \"\"\"Evaluate retrieval: Recall@5/10/20, MRR, before and after reranking.\"\"\"\n",
    "    results_per_row = []\n",
    "\n",
    "    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc='Evaluating'):\n",
    "        question = row['question']\n",
    "        gold_chunk_id = str(row['chunk_id'])\n",
    "\n",
    "        retrieved = retrieve_chunks(\n",
    "            question, embed_model, faiss_index,\n",
    "            all_chunks, chunk_metadata, top_k=top_k_retrieve\n",
    "        )\n",
    "        retrieved_ids = [r['chunk_id'] for r in retrieved]\n",
    "\n",
    "        reranked = rerank_results(question, retrieved, reranker_model, top_n=top_n_rerank)\n",
    "        reranked_ids = [r['chunk_id'] for r in reranked]\n",
    "\n",
    "        hit_at_5  = 1 if gold_chunk_id in retrieved_ids[:5]  else 0\n",
    "        hit_at_10 = 1 if gold_chunk_id in retrieved_ids[:10] else 0\n",
    "        hit_at_20 = 1 if gold_chunk_id in retrieved_ids[:20] else 0\n",
    "\n",
    "        if gold_chunk_id in retrieved_ids:\n",
    "            mrr = 1.0 / (retrieved_ids.index(gold_chunk_id) + 1)\n",
    "        else:\n",
    "            mrr = 0.0\n",
    "\n",
    "        hit_at_5_reranked = 1 if gold_chunk_id in reranked_ids else 0\n",
    "        if gold_chunk_id in reranked_ids:\n",
    "            mrr_reranked = 1.0 / (reranked_ids.index(gold_chunk_id) + 1)\n",
    "        else:\n",
    "            mrr_reranked = 0.0\n",
    "\n",
    "        results_per_row.append({\n",
    "            'qa_id':              row.get('qa_id', ''),\n",
    "            'question_type':      row.get('question_type', ''),\n",
    "            'doc_type':           row.get('doc_type', ''),\n",
    "            'coverage_status':    row.get('coverage_status', ''),\n",
    "            'hit_at_5':           hit_at_5,\n",
    "            'hit_at_10':          hit_at_10,\n",
    "            'hit_at_20':          hit_at_20,\n",
    "            'mrr':                mrr,\n",
    "            'hit_at_5_reranked':  hit_at_5_reranked,\n",
    "            'mrr_reranked':       mrr_reranked,\n",
    "        })\n",
    "\n",
    "    results_df = pd.DataFrame(results_per_row)\n",
    "    summary = {\n",
    "        'Recall@5 (FAISS)':    results_df['hit_at_5'].mean(),\n",
    "        'Recall@10 (FAISS)':   results_df['hit_at_10'].mean(),\n",
    "        'Recall@20 (FAISS)':   results_df['hit_at_20'].mean(),\n",
    "        'MRR (FAISS)':         results_df['mrr'].mean(),\n",
    "        'Recall@5 (reranked)': results_df['hit_at_5_reranked'].mean(),\n",
    "        'MRR (reranked)':      results_df['mrr_reranked'].mean(),\n",
    "    }\n",
    "    return summary, results_df\n",
    "\n",
    "\n",
    "print('Helper functions defined.')",
]))

# ── Now add original cells 63-75 (Phase 6 + Phase 7) ────────────────────────
for c in cells[63:]:
    new_cell = copy.deepcopy(c)
    if new_cell["cell_type"] == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    nb_b["cells"].append(new_cell)

with open(OUT_B, "w", encoding="utf-8") as f:
    json.dump(nb_b, f, indent=1, ensure_ascii=False)

print(f"Notebook B: {len(nb_b['cells'])} cells -> {OUT_B}")
