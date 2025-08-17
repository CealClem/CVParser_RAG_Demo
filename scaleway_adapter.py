import asyncio

import json
from openai import AsyncOpenAI
from interfaces import LLMClientInterface

class ScalewayLLMClient(LLMClientInterface):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("SCALEWAY_API_KEY"),
            base_url="https://api.scaleway.ai/v1"
        )

    async def extract_data_async(self, prompt: str) -> dict:
        response = await self.client.chat.completions.create(
            model="qwen2.5-coder-32b-instruct",
            messages=[
                {"role": "system",
                 "content": "You are an accurate system that extracts all relevant data from CVs, including identity, education, employment, skills, languages and hobbies. Follow the provided structure strictly and return only a valid JSON. You have knowledge of the French school system so you'll know how to fill in the education level part : eg. Licence=BAC+3, Master=BAC+5, PHD=BAC+8 etc."},
                {"role": "user", "content": prompt}
            ],
            functions=[
                {
                    "name": "extract_cv_data",
                    "description": "Extracts structured data from CVs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identity": {
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
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_date": {"type": "string"},
                                        "end_date": {"type": "string"},
                                        "title": {"type": "string"},
                                        "employer": {"type": "string"},
                                        "description": {"type": "string"}
                                    }
                                }
                            },
                            "educations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_date": {"type": "string"},
                                        "end_date": {"type": "string"},
                                        "title": {"type": "string"},
                                        "institution": {"type": "string"},
                                        "level_code": {"type": "string"},
                                        "description": {"type": "string"}
                                    }
                                }
                            },
                            "skills": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "level_code": {"type": "string"}
                                    }
                                }
                            },
                            "languages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "language_code": {"type": "string"},
                                        "level_code": {"type": "string"}
                                    }
                                }
                            },
                            "hobbies": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["identity", "employments", "educations", "skills", "languages", "hobbies"]
                    }
                }
            ],
            function_call={"name": "extract_cv_data"}
        )
        content = response.choices[0].message.content
        return json.loads(content.replace('```json', '').replace('```', ''))

    def extract_data(self, prompt: str) -> dict:
        """Synchronous wrapper for extract_data_async"""
        return asyncio.run(self.extract_data_async(prompt))