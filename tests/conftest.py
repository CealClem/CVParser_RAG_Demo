import pytest
from typing import Any, Dict

import main as main_app


class _StubLLMClient:
	def generate_prompt(self, cv_text: str, prompt_file: str) -> str:
		return cv_text

	def extract_data(self, prompt: str) -> Dict[str, Any]:
		# Minimal structured data to satisfy main.result rendering
		return {
			"educations": [{"title": "Degree", "institution": "Uni", "start_date": "2020-01-01", "end_date": "2021-01-01"}],
			"employments": [{"title": "Engineer", "employer": "Acme", "start_date": "2022-01-01", "end_date": None}],
			"languages": [{"language": "English", "level": "Fluent"}],
			"identity": [{"first_name": "Jane", "last_name": "Doe"}],
			"hobbies": [{"name": "Reading"}],
			"skills": [{"name": "Python"}],
			"certifications": [{"name": "Cert", "date": "2023-01-01"}],
		}


@pytest.fixture()
def app(monkeypatch):
	# Ensure testing mode
	main_app.app.config.update({"TESTING": True, "SECRET_KEY": "test-secret"})
	# Provide predictable llm client stub for /result route tests
	monkeypatch.setattr(main_app, "llm_client", _StubLLMClient(), raising=True)
	return main_app.app


@pytest.fixture()
def client(app):
	return app.test_client()