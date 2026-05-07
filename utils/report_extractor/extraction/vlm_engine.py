# midas_extraction/extraction/vlm_engine.py

import json
import ollama
from PIL import Image
from typing import Type
from pydantic import BaseModel


class VLMEngine:
    """
    Vision-Language extraction engine using local Ollama Qwen2.5-VL.

    This module:
    - sends image + prompt to Ollama
    - receives structured JSON
    - returns parsed dict
    """

    def __init__(
        self,
        model_name: str = "qwen2.5vl",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature

    def extract(
        self,
        image: Image.Image,
        schema: Type[BaseModel],
        system_prompt: str | None = None,
    ) -> dict:
        """
        Extract structured medical report.

        Args:
            image: Preprocessed PIL image
            schema: Pydantic schema class
            system_prompt: optional system prompt

        Returns:
            dict matching schema
        """

        # Convert schema to JSON schema string
        json_schema = schema.model_json_schema()
        print(f"Using JSON schema for prompt:\n{json.dumps(json_schema, indent=2)}")  # Debug print

        prompt = f"""
You are a medical report extraction system.

Extract information from the medical report image.

Return ONLY valid JSON matching this schema:

{json.dumps(json_schema, indent=2)}

Rules:
- No explanations
- No markdown
- No extra text
- Only valid JSON
"""

        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt

        # Convert PIL image to bytes
        import io
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Call Ollama
        response = ollama.chat(
            model=self.model_name,
            options={
                "temperature": self.temperature,
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
        )

        content = response["message"]["content"]

        # Parse JSON safely
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from model:\n{content}")

        return parsed