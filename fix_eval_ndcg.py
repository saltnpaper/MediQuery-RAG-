"""Fix CERerankingEvaluator: skip samples with empty negatives to avoid NDCG error."""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK = 'Fine tuning codex v3.ipynb'

with open(NOTEBOOK, encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][48]['source'] = [
    "# Build eval samples for CERerankingEvaluator (using fine-tuned negatives)\n",
    "# Skip samples with no negatives — CERerankingEvaluator computes NDCG internally\n",
    "# and requires at least 2 documents (1 positive + 1 negative) per query.\n",
    "reranker_eval_samples = []\n",
    "skipped = 0\n",
    "for pair in val_set:\n",
    "    if not pair.get('ft_negatives'):\n",
    "        skipped += 1\n",
    "        continue\n",
    "    reranker_eval_samples.append({\n",
    "        'query':    pair['query'],\n",
    "        'positive': [pair['positive']],\n",
    "        'negative': pair['ft_negatives']\n",
    "    })\n",
    "\n",
    "reranker_evaluator = CERerankingEvaluator(\n",
    "    reranker_eval_samples,\n",
    "    name='mediquery-reranker-val'\n",
    ")\n",
    "\n",
    'print(f"Reranker evaluator: {len(reranker_eval_samples)} samples (skipped {skipped} with no negatives)")\n',
]
nb['cells'][48]['outputs'] = []
nb['cells'][48]['execution_count'] = None

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Fixed {NOTEBOOK}: cell 48 — filter out samples with empty negatives')
