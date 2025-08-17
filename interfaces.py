from abc import ABC, abstractmethod
import os

class LLMClientInterface(ABC):

    def generate_prompt(self, cv_text: str, prompt_file=None) -> str:
        # Construct the full path to the prompt file
        prompt_path = os.path.join("prompt_versions", prompt_file)

        # Load the prompt from the file
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        # Replace the CV text placeholder using string replacement instead of .format()
        # This avoids conflicts with curly braces in the JSON examples
        return prompt_template.replace("{cv_text}", cv_text)

    @abstractmethod
    def extract_data(self, prompt: str) -> dict:
        pass
