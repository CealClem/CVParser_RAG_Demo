# ollama_adapter.py

import subprocess
import json
from interfaces import LLMClientInterface
from langchain_community.llms import Ollama

import ollama

# Needs work.
# for the moment this model does not respect the format and takes ages to run.

class OllamaLLMClient(LLMClientInterface):
    def __init__(self, model_name="mistral"):
        self.model_name = model_name

    def extract_data(self, prompt: str) -> dict:
        try:
            llm = Ollama(model="mistral")
            output = llm.invoke(prompt)
            return json.loads(output.replace('```json', '').replace('```', ''))
        except subprocess.CalledProcessError as e:
            raise Exception(f"Ollama failed: {e.stderr}")
