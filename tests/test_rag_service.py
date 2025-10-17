import os
import math
import pytest

from rag_service import RAGService


def test_chunk_document_basic():
	service = RAGService()
	text = "one two three four five six seven eight nine ten"
	chunks = service.chunk_document(text, chunk_size=4, overlap=1)
	assert len(chunks) > 0
	# Check metadata presence
	for ch in chunks:
		assert set(["id", "text", "start_word", "end_word", "word_count", "start_char", "end_char"]).issubset(ch.keys())


def test_embeddings_fallback_without_api_key(monkeypatch):
	# Ensure no OpenAI key so TF-IDF is used
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)
	service = RAGService()
	text = "Python developer with data experience. Loves Python and data."
	chunks = service.chunk_document(text, chunk_size=5, overlap=2)
	embeddings = service.generate_embeddings(chunks)
	assert len(embeddings) == len(chunks)
	assert all(isinstance(v, float) for v in embeddings[0])
	# Embedding previews should exist in summary
	summary = service.get_pipeline_summary()
	assert summary["embeddings_generated"] == len(chunks)
	assert isinstance(summary["embedding_previews"], list)
	if summary["embedding_previews"]:
		assert "values" in summary["embedding_previews"][0]


def test_retrieve_relevant_chunks(monkeypatch):
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)
	service = RAGService()
	text = "Python developer. Data scientist. Experience with Flask and pandas."
	chunks = service.chunk_document(text, chunk_size=4, overlap=1)
	service.generate_embeddings(chunks)
	top = service.retrieve_relevant_chunks("What experience with Python?", top_k=2)
	assert len(top) <= 2
	for item in top:
		assert "similarity_score" in item and isinstance(item["similarity_score"], float)
		assert "rank" in item and isinstance(item["rank"], int)


def test_pipeline_summary_fields(monkeypatch):
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)
	service = RAGService()
	chunks = service.chunk_document("alpha beta gamma delta epsilon zeta eta theta", chunk_size=3, overlap=1)
	service.generate_embeddings(chunks)
	summary = service.get_pipeline_summary()
	for key in ["total_chunks", "chunk_size_avg", "embeddings_generated", "embedding_dimension", "pipeline_steps", "embedding_previews"]:
		assert key in summary
	assert isinstance(summary["pipeline_steps"], list)