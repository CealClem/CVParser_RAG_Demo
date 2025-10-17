import json
import pytest


def test_index_route(client):
	resp = client.get("/")
	assert resp.status_code == 200
	assert b"CV Extract" in resp.data


def test_rag_process_requires_text(client):
	resp = client.post("/rag_process", data={"cv_text": ""})
	assert resp.status_code == 200
	data = resp.get_json()
	assert "error" in data


def test_rag_process_success(client):
	cv_text = "Python developer with 5 years of experience in Flask and data science."
	resp = client.post("/rag_process", data={"cv_text": cv_text})
	assert resp.status_code == 200
	data = resp.get_json()
	assert data.get("success") is True
	assert isinstance(data.get("chunks"), list)
	assert isinstance(data.get("pipeline_summary"), dict)


def test_rag_query_requires_query(client):
	resp = client.post("/rag_query", data={"query": ""})
	assert resp.status_code == 200
	data = resp.get_json()
	assert "error" in data


def test_rag_query_requires_cv_in_session(client):
	resp = client.post("/rag_query", data={"query": "What skills?"})
	data = resp.get_json()
	assert "No CV loaded" in data.get("error", "")


def test_rag_query_success(client):
	# First seed the session with a CV via rag_process
	cv_text = "Flask developer skilled in Python and pandas."
	_ = client.post("/rag_process", data={"cv_text": cv_text})
	resp = client.post("/rag_query", data={"query": "What skills?"})
	assert resp.status_code == 200
	data = resp.get_json()
	assert data.get("success") is True
	assert isinstance(data.get("retrieved_chunks"), list)
	assert "answer" in data


def test_result_route_renders_html(client):
	cv_text = "Python developer at Acme Corp."
	resp = client.post("/result", data={"cv": cv_text})
	assert resp.status_code == 200
	# Expect HTML response containing table markup from pandas to_html
	assert b"table" in resp.data