import json
import os
import outlines
from interfaces import LLMClientInterface

class OutlinesLLMClient(LLMClientInterface):
    def __init__(self):
        # Initialize the Outlines model
        self.model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
        # Define the schema for JSON generation
        self.schema = '''{
            "title": "CVData",
            "type": "object",
            "properties": {
                "identity": {
                    "title": "Identity",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string"},
                            "last_name": {"type": "string"},
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                            "address": {"type": "string"},
                            "postal_code": {"type": "string"},
                            "town": {"type": "string"},
                            "country": {"type": "string"}
                        },
                        "required": ["first_name", "last_name"]
                    }
                },
                "employments": {
                    "title": "Employments",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"},
                            "title": {"type": "string"},
                            "employer": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["start_date", "end_date", "title"]
                    }
                },
                "educations": {
                    "title": "Educations",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"},
                            "title": {"type": "string"},
                            "institution": {"type": "string"},
                            "level_code": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["start_date", "end_date", "title", "institution"]
                    }
                },
                "skills": {
                    "title": "Skills",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "level_code": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                },
                "hobbies": {
                    "title": "Hobbies",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["identity", "employments", "educations", "skills", "hobbies"]
        }'''

        self.generator = outlines.generate.json(self.model, self.schema)

    def generate_prompt(self, cv_text, prompt_file):
        # Construct the full path to the prompt file
        prompt_path = os.path.join("prompt_versions", prompt_file)

        # Load the prompt from the file
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        # Integrate the CV text into the prompt template
        return prompt_template.format(cv_text=cv_text)

    def extract_data(self, prompt: str) -> dict:
        # Use the Outlines generator with the schema and prompt
        response = self.generator(prompt)

        # Validate and parse the JSON response
        try:
            extracted_data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        return extracted_data

