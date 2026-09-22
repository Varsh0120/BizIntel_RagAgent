"""
Tests for BizIntel's core logic.

Run with:
    pytest

These focus on the pure, side-effect-free functions (chunking strategy,
scoring) so they run offline, fast, and without needing real Supabase/model
credentials. The health-check test exercises the actual FastAPI app.

For a full end-to-end test (upload -> ask -> real model answer), you'd
additionally need real API keys and network access — that's intentionally
kept separate so `pytest` works in CI without secrets.
"""

import os

os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
os.environ.setdefault("AUTH_API_KEY", "dummy_key")
os.environ.setdefault("DATABASE_API_KEY", "dummy_key")
os.environ.setdefault("GROQ_API_KEY", "dummy_key")

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
import numpy as np

import main


client = TestClient(main.app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_markdown_chunking_splits_by_section():
    md_text = b"# Report\n\n## Revenue\nRevenue grew 12 percent.\n\n## Risks\nSupply delays continue.\n"
    chunks = main._extract_chunks_from_upload(md_text, "report.md")

    sections = {c["metadata"].get("section") for c in chunks}
    assert "Revenue" in sections
    assert "Risks" in sections
    # Each section's content should not bleed into the other
    revenue_chunk = next(c for c in chunks if c["metadata"].get("section") == "Revenue")
    assert "Supply delays" not in revenue_chunk["text"]


def test_csv_rows_are_not_split_when_short():
    csv_text = b"text\nRevenue grew 12 percent.\nHeadcount rose 5 percent.\n"
    chunks = main._extract_chunks_from_upload(csv_text, "data.csv")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["row_index"] == 0
    assert chunks[1]["metadata"]["row_index"] == 1


def test_plain_text_falls_back_to_prose_splitting():
    text = b"Just a short plain-text note with no headers."
    chunks = main._extract_chunks_from_upload(text, "note.txt")

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "note.txt"


def test_keyword_overlap_favors_relevant_chunk_over_drifted_chunk():
    query_tokens = main._tokenize("How did Q3 revenue grow?")

    relevant = "Revenue grew 12 percent in Q3 driven by enterprise deals."
    drifted = "Q3 headcount increased across all departments this year."

    relevant_score = main._keyword_overlap_score(query_tokens, relevant)
    drifted_score = main._keyword_overlap_score(query_tokens, drifted)

    assert relevant_score > drifted_score


def test_distance_to_similarity_is_monotonically_decreasing():
    close = main._distance_to_similarity(0.1)
    far = main._distance_to_similarity(0.9)
    assert close > far
    assert 0.0 <= far <= 1.0
    assert 0.0 <= close <= 1.0


def test_upload_requires_authentication():
    response = client.post(
        "/upload-document",
        files={"file": ("empty.txt", b"")},
    )
    assert response.status_code == 401


def test_reranker_uses_thresholded_semantic_fallback(monkeypatch):
    class LowScoringReranker:
        def predict(self, pairs, **kwargs):
            return [0.001 for _ in pairs]

    monkeypatch.setattr(main, "_get_reranker", lambda: LowScoringReranker())
    candidates = [{
        "text": "REGISTER NUMBER NAME 162 Alice 163 Bob",
        "metadata": {"document_id": "doc-1", "user_id": "user-1"},
        "similarity": 0.32,
        "keyword_overlap": 0.0,
    }]

    ranked = main._rerank_candidates("How many names are in this list?", candidates)

    assert len(ranked) == 1
    assert ranked[0]["metadata"]["document_id"] == "doc-1"


def test_reranker_returns_empty_when_all_thresholds_fail(monkeypatch):
    class LowScoringReranker:
        def predict(self, pairs, **kwargs):
            return [0.001 for _ in pairs]

    monkeypatch.setattr(main, "_get_reranker", lambda: LowScoringReranker())
    candidates = [{
        "text": "Unrelated content",
        "metadata": {"document_id": "doc-2", "user_id": "user-1"},
        "similarity": 0.05,
        "keyword_overlap": 0.0,
    }]

    assert main._rerank_candidates("How many names?", candidates) == []


def test_document_id_is_validated():
    with pytest.raises(ValidationError):
        main.OrchestrationPayload(
            prompt="Question",
            document_id="not-a-uuid",
        )


def test_retrieval_scopes_vectors_to_user_and_document(monkeypatch):
    captured = {}

    class FakeEmbeddingModel:
        def encode(self, text, **kwargs):
            return np.array([0.1, 0.2, 0.3])

    class FakeCollection:
        def query(self, **kwargs):
            captured.update(kwargs)
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    monkeypatch.setattr(main, "_get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(main, "collection", FakeCollection())

    result = main._retrieve_candidates(
        "How many names are in this list?",
        "user-1",
        "document-1",
    )

    assert result == []
    assert captured["where"] == {
        "$and": [
            {"user_id": "user-1"},
            {"document_id": "document-1"},
        ]
    }


def test_vector_deletion_only_removes_owned_document(monkeypatch):
    deleted = []

    class FakeCollection:
        def get(self, **kwargs):
            assert kwargs["where"] == {"user_id": "user-1"}
            return {
                "ids": ["a", "b", "c"],
                "metadatas": [
                    {"document_id": "document-1"},
                    {"document_id": "document-2"},
                    {"document_id": "document-1"},
                ],
            }

        def delete(self, **kwargs):
            deleted.extend(kwargs["ids"])

    monkeypatch.setattr(main, "collection", FakeCollection())

    main._delete_document_vectors("user-1", "document-1")

    assert deleted == ["a", "c"]
