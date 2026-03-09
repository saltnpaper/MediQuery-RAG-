"""Update Phase 4 in 4_Fine_Tuning.ipynb to re-mine hard negatives from the fine-tuned FAISS index."""
import json

NOTEBOOK = '4_Fine_Tuning.ipynb'

with open(NOTEBOOK, encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 41 (markdown): update description ──
nb['cells'][41]['source'] = [
    '### Step 4.1 \u2014 Re-mine hard negatives from the fine-tuned FAISS index\n',
    '\n',
    'The bi-encoder hard negatives mined in Phase 1 came from the **pre-trained** index. Now that we have a fine-tuned index, we re-mine harder negatives \u2014 passages the fine-tuned bi-encoder ranks highly but that come from a different source than the gold chunk. These are more challenging for the reranker to distinguish, leading to better training signal.\n',
    '\n',
    'Cross-encoders are trained on `(query, passage, label)` pairs where `label = 1.0` means relevant and `label = 0.0` means irrelevant.',
]

# ── Cell 42 (code): re-mine from fine-tuned index + build reranker examples ──
nb['cells'][42]['source'] = [
    '# Re-mine hard negatives using the fine-tuned bi-encoder + fine-tuned FAISS index\n',
    'print("Re-mining hard negatives from the fine-tuned FAISS index...\\n")\n',
    '\n',
    '# Batch-encode all training queries with the fine-tuned model\n',
    'train_queries = [p[\'query\'] for p in train_set]\n',
    'ft_query_embeddings = finetuned_embed_model.encode(\n',
    '    train_queries,\n',
    '    batch_size=256,\n',
    '    normalize_embeddings=True,\n',
    '    show_progress_bar=True,\n',
    '    convert_to_numpy=True\n',
    ').astype(\'float32\')\n',
    '\n',
    '# Search fine-tuned FAISS index\n',
    'TOP_K_MINE_FT = 20\n',
    'scores_ft, indices_ft = finetuned_index.search(ft_query_embeddings, TOP_K_MINE_FT)\n',
    '\n',
    '# Assign hard negatives from the fine-tuned index\n',
    'no_negative_count = 0\n',
    'for i, pair in enumerate(tqdm(train_set, desc="Assigning FT hard negatives")):\n',
    '    gold_source = pair[\'source_id\']\n',
    '    negative_text = None\n',
    '\n',
    '    for idx in indices_ft[i]:\n',
    '        if idx == -1:\n',
    '            continue\n',
    '        meta = chunk_metadata[idx]\n',
    '        if str(meta[\'source_id\']) != gold_source:\n',
    '            negative_text = all_chunks[idx][\'text\']\n',
    '            break\n',
    '\n',
    '    if negative_text is None:\n',
    '        fallback_idx = indices_ft[i][-1]\n',
    '        if fallback_idx != -1:\n',
    '            negative_text = all_chunks[fallback_idx][\'text\']\n',
    '        else:\n',
    '            negative_text = pair[\'negative\']  # keep original if no match\n',
    '        no_negative_count += 1\n',
    '\n',
    '    pair[\'ft_negative\'] = negative_text\n',
    '\n',
    'print(f"\\nRe-mined hard negatives for {len(train_set):,} training pairs")\n',
    'print(f"Pairs where no distinct-source negative was found: {no_negative_count}")\n',
    '\n',
    '# Also re-mine for validation set\n',
    'val_queries = [p[\'query\'] for p in val_set]\n',
    'ft_val_embeddings = finetuned_embed_model.encode(\n',
    '    val_queries,\n',
    '    batch_size=256,\n',
    '    normalize_embeddings=True,\n',
    '    show_progress_bar=True,\n',
    '    convert_to_numpy=True\n',
    ').astype(\'float32\')\n',
    '\n',
    'scores_val_ft, indices_val_ft = finetuned_index.search(ft_val_embeddings, TOP_K_MINE_FT)\n',
    '\n',
    'for i, pair in enumerate(tqdm(val_set, desc="Assigning FT val hard negatives")):\n',
    '    gold_source = pair[\'source_id\']\n',
    '    negative_text = None\n',
    '    for idx in indices_val_ft[i]:\n',
    '        if idx == -1:\n',
    '            continue\n',
    '        meta = chunk_metadata[idx]\n',
    '        if str(meta[\'source_id\']) != gold_source:\n',
    '            negative_text = all_chunks[idx][\'text\']\n',
    '            break\n',
    '    if negative_text is None:\n',
    '        negative_text = pair[\'negative\']\n',
    '    pair[\'ft_negative\'] = negative_text\n',
    '\n',
    '# Build reranker training examples using fine-tuned negatives\n',
    'reranker_train_examples = []\n',
    'for pair in train_set:\n',
    '    reranker_train_examples.append(\n',
    '        InputExample(texts=[pair[\'query\'], pair[\'positive\']], label=1.0)\n',
    '    )\n',
    '    reranker_train_examples.append(\n',
    '        InputExample(texts=[pair[\'query\'], pair[\'ft_negative\']], label=0.0)\n',
    '    )\n',
    '\n',
    'print(f"\\nReranker training examples: {len(reranker_train_examples):,}")\n',
    '\n',
    '# Validation data using fine-tuned negatives\n',
    'reranker_val_examples = []\n',
    'for pair in val_set:\n',
    '    reranker_val_examples.append(\n',
    '        InputExample(texts=[pair[\'query\'], pair[\'positive\']], label=1.0)\n',
    '    )\n',
    '    reranker_val_examples.append(\n',
    '        InputExample(texts=[pair[\'query\'], pair[\'ft_negative\']], label=0.0)\n',
    '    )\n',
    '\n',
    'print(f"Reranker validation examples: {len(reranker_val_examples):,}")',
]
nb['cells'][42]['outputs'] = []
nb['cells'][42]['execution_count'] = None

# ── Cell 46 (code): update evaluator to use ft_negative ──
nb['cells'][46]['source'] = [
    '# Build eval samples for CERerankingEvaluator (using fine-tuned negatives)\n',
    'reranker_eval_samples = []\n',
    'for pair in val_set:\n',
    '    reranker_eval_samples.append({\n',
    "        'query':    pair['query'],\n",
    "        'positive': [pair['positive']],\n",
    "        'negative': [pair['ft_negative']]\n",
    '    })\n',
    '\n',
    'reranker_evaluator = CERerankingEvaluator(\n',
    '    reranker_eval_samples,\n',
    "    name='mediquery-reranker-val'\n",
    ')\n',
    '\n',
    'print(f"Reranker evaluator: {len(reranker_eval_samples)} evaluation samples")',
]
nb['cells'][46]['outputs'] = []
nb['cells'][46]['execution_count'] = None

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Done. Updated cells 41, 42, and 46.')
