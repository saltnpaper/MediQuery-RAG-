from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


FALLBACK_TEXT = "Insufficient evidence in retrieved documents."
DEFAULT_MODEL = os.getenv("MEDIQUERY_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
TOKEN_RE = re.compile(r"[a-zA-Z0-9]{2,}")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
    "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    source_id: str
    doc_type: str
    states: list[str]
    text: str
    score: float


class RAGService:
    def __init__(self, data_path: str | os.PathLike[str]) -> None:
        self.data_path = Path(data_path)
        self.chunks = self._load_chunks()
        self.doc_freq = self._build_doc_frequency()
        self.artifact_dir = Path(os.getenv("MEDIQUERY_ARTIFACT_DIR", self.data_path.parent / "faiss_index"))
        self.use_finetuned = os.getenv("MEDIQUERY_USE_FINETUNED", "false").lower() == "true"
        self.embed_model_name = os.getenv("MEDIQUERY_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.faiss_index = None
        self.embed_model = None
        self.metadata: list[dict[str, Any]] = []
        self.backend_mode = "lexical"
        self._load_vector_stack()

    def _load_chunks(self) -> list[dict[str, Any]]:
        with self.data_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _load_vector_stack(self) -> None:
        if faiss is None or SentenceTransformer is None:
            return

        index_name = "medicare_finetuned.index" if self.use_finetuned else "medicare.index"
        index_path = self.artifact_dir / index_name
        metadata_path = self.artifact_dir / "chunk_metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return

        self.faiss_index = faiss.read_index(str(index_path))
        with metadata_path.open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.backend_mode = "faiss"

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in TOKEN_RE.findall(text)
            if token.lower() not in STOP_WORDS
        ]

    def _build_doc_frequency(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for chunk in self.chunks:
            for token in set(self._tokenize(chunk.get("text", ""))):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _extract_state(self, query: str) -> str | None:
        upper = query.upper()
        tokens = re.findall(r"\b[A-Z]{2}\b", upper)
        for token in tokens:
            if token in STATE_CODES:
                return token
        return None

    def _embed_query(self, query: str) -> np.ndarray:
        if self.embed_model is None:
            raise RuntimeError("Embedding model not loaded.")
        return self.embed_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def _retrieve_faiss(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.faiss_index is None:
            return []

        query_vector = self._embed_query(query)
        candidate_k = min(max(top_k * 5, 20), len(self.chunks))
        scores, indices = self.faiss_index.search(np.array([query_vector]), candidate_k)

        state = self._extract_state(query)
        results: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            chunk = self.chunks[idx]
            states = list(meta.get("states", chunk.get("states", ["ALL"])))

            if state and "ALL" not in states and state not in states:
                continue

            chunk_id = f"{chunk.get('source_id', 'unknown')}#{chunk.get('chunk_idx', 0)}"
            if chunk_id in seen_chunk_ids:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    title=str(meta.get("title", chunk.get("title", "Untitled"))),
                    source_id=str(meta.get("source_id", chunk.get("source_id", "unknown"))),
                    doc_type=str(meta.get("type", chunk.get("type", "UNKNOWN"))),
                    states=states,
                    text=str(chunk.get("text", "")),
                    score=float(score),
                )
            )
            seen_chunk_ids.add(chunk_id)

            if len(results) == top_k:
                break

        return results

    def _score_chunk(self, query_tokens: list[str], chunk: dict[str, Any]) -> float:
        chunk_tokens = self._tokenize(chunk.get("text", ""))
        title_tokens = self._tokenize(chunk.get("title", ""))
        source_tokens = self._tokenize(str(chunk.get("source_id", "")))
        if not chunk_tokens:
            return 0.0

        chunk_tf: dict[str, int] = {}
        for token in chunk_tokens:
            chunk_tf[token] = chunk_tf.get(token, 0) + 1

        title_tf: dict[str, int] = {}
        for token in title_tokens:
            title_tf[token] = title_tf.get(token, 0) + 1

        source_tf: dict[str, int] = {}
        for token in source_tokens:
            source_tf[token] = source_tf.get(token, 0) + 1

        total_docs = len(self.chunks)
        score = 0.0
        for token in query_tokens:
            idf = math.log((1 + total_docs) / (1 + self.doc_freq.get(token, 0))) + 1
            score += chunk_tf.get(token, 0) * idf
            score += title_tf.get(token, 0) * idf * 4
            score += source_tf.get(token, 0) * idf * 2

        text_lower = chunk.get("text", "").lower()
        title_lower = chunk.get("title", "").lower()
        query_text = " ".join(query_tokens)
        if query_text and query_text in text_lower:
            score += 10
        if query_text and query_text in title_lower:
            score += 14

        overlap = len(set(query_tokens) & set(title_tokens))
        score += overlap * 2

        return score / math.sqrt(len(chunk_tokens))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.backend_mode == "faiss":
            faiss_results = self._retrieve_faiss(query, top_k)
            if faiss_results:
                return faiss_results

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        state = self._extract_state(query)
        results: list[RetrievedChunk] = []

        for chunk in self.chunks:
            states = chunk.get("states", ["ALL"])
            if state and "ALL" not in states and state not in states:
                continue

            score = self._score_chunk(query_tokens, chunk)
            if score <= 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=f"{chunk.get('source_id', 'unknown')}#{chunk.get('chunk_idx', 0)}",
                    title=str(chunk.get("title", "Untitled")),
                    source_id=str(chunk.get("source_id", "unknown")),
                    doc_type=str(chunk.get("type", "UNKNOWN")),
                    states=list(states),
                    text=str(chunk.get("text", "")),
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        selected = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:5]
        context_blocks = []
        for index, chunk in enumerate(selected, start=1):
            context_blocks.append(
                f"[C{index}] title={chunk.title}; source={chunk.source_id}; url=; score={chunk.score:.4f}\n"
                f"evidence: {chunk.text}"
            )

        return (
            "You are a healthcare QA assistant.\n"
            "Use ONLY the provided evidence.\n"
            "Do NOT invent facts.\n"
            f"If evidence is insufficient, output exactly: {FALLBACK_TEXT}\n\n"
            "Return a JSON object with exactly these keys:\n"
            '- "answer" (string, 2-4 patient-friendly sentences with inline citations like [C1], [C2])\n'
            '- "citations" (array of objects)\n\n'
            "Each citation object must have exactly these keys:\n"
            '- "marker" (string)\n'
            '- "title" (string)\n'
            '- "source" (string)\n'
            '- "url" (string)\n\n'
            "Do not return any extra text outside the JSON.\n\n"
            f"User question:\n{question}\n\n"
            f"Context:\n{'\n\n'.join(context_blocks)}\n"
        )

    def _call_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent"
            f"?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 700},
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        candidates = body.get("candidates", [])
        if not candidates:
            return FALLBACK_TEXT

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(part.get("text", "") for part in parts).strip()
        return answer or FALLBACK_TEXT

    def _build_extractive_answer(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return FALLBACK_TEXT

        answer_lines = [
            "Here is the strongest evidence I found in the indexed healthcare documents:",
        ]
        for index, chunk in enumerate(chunks, start=1):
            snippet = chunk.text.replace("\n", " ").strip()
            answer_lines.append(
                f"[C{index}] {snippet[:280]}{'...' if len(snippet) > 280 else ''}"
            )
        answer_lines.append("")
        answer_lines.append("Citations")
        for index, chunk in enumerate(chunks, start=1):
            answer_lines.append(f"[C{index}] {chunk.title} | {chunk.source_id}")
        return "\n".join(answer_lines)

    def _evidence_is_sufficient(self, chunks: list[RetrievedChunk], min_score: float = 0.2, min_chunks: int = 2) -> bool:
        if len(chunks) < min_chunks:
            return False
        strong = [chunk for chunk in chunks if float(chunk.score) >= min_score]
        return len(strong) >= min_chunks

    def _synthesize_answer(self, question: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        if not self._evidence_is_sufficient(chunks):
            return {"answer": FALLBACK_TEXT, "prompt": "", "citations": []}

        prompt = self._build_prompt(question, chunks)
        raw_response = self._call_gemini(prompt)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "answer": raw_response,
                "prompt": prompt,
                "citations": [],
            }

        return {
            "answer": parsed.get("answer", FALLBACK_TEXT),
            "prompt": prompt,
            "citations": parsed.get("citations", []),
        }

    def answer(self, question: str) -> dict[str, Any]:
        chunks = self.retrieve(question, top_k=5)
        if len(chunks) < 2:
            return {
                "answer": FALLBACK_TEXT,
                "citations": [],
                "retrieved_chunks": [],
                "mode": "fallback",
                "llm_enabled": bool(os.getenv("GEMINI_API_KEY", "").strip()),
                "llm_error": "Insufficient retrieved evidence.",
            }

        llm_error = None
        llm_citations: list[dict[str, Any]] = []
        try:
            synthesis = self._synthesize_answer(question, chunks)
            answer = synthesis["answer"]
            llm_citations = synthesis.get("citations", [])
            mode = "rag"
        except Exception as exc:
            answer = self._build_extractive_answer(chunks)
            mode = "extractive"
            llm_error = str(exc)

        return {
            "answer": answer,
            "citations": llm_citations or [
                {
                    "id": f"C{index}",
                    "title": chunk.title,
                    "source_id": chunk.source_id,
                    "type": chunk.doc_type,
                    "states": chunk.states,
                    "score": round(chunk.score, 4),
                }
                for index, chunk in enumerate(chunks, start=1)
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source_id": chunk.source_id,
                    "type": chunk.doc_type,
                    "states": chunk.states,
                    "score": round(chunk.score, 4),
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
            "mode": mode,
            "retrieval_backend": self.backend_mode,
            "llm_enabled": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "llm_error": llm_error,
        }
