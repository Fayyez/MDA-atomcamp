# midas_extraction/extraction/vlm_engine.py

import json
import base64
import io
import os
from openai import OpenAI
from PIL import Image
from typing import Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class VLMEngine:
    """
    Vision-Language extraction engine using OpenAI GPT-4o.

    This module:
    - sends image + prompt to OpenAI
    - receives structured JSON
    - returns parsed dict
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI()  # Expects OPENAI_API_KEY environment variable

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

        # Convert PIL image to base64
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        # Call OpenAI
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )

        content = response.choices[0].message.content

        # Parse JSON safely
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from model:\n{content}")

        return parsed