"""
Medical Report Extractor Module.

This module orchestrates the extraction of structured clinical data from images of 
medical reports (scans, faxes, or clinical photos). It uses document preprocessing 
and a Vision-Language Model (VLM) engine for JSON extraction, ensuring strict 
validation against predefined clinical schemas (Pydantic).

Usage Example:
-------------
import json
from PIL import Image

from report_extractor.preprocessing.image_preprocessor import ImagePreprocessor
from report_extractor.extraction.vlm_engine import VLMEngine
from report_extractor.validation.schema_validator import SchemaValidator, MedicalReport

def extract_structured_data_from_report(image_path: str) -> dict:
    # 1. Load image
    img = Image.open(image_path).convert("RGB")
    
    # 2. Preprocess report image for text clarity
    preprocessor = ImagePreprocessor()
    clean_img = preprocessor.preprocess(img)
    
    # 3. Extract JSON using VLM mapped to the MedicalReport schema
    vlm = VLMEngine()
    raw_json = vlm.extract(clean_img, MedicalReport)
    
    # 4. Validate output schema using Pydantic
    validated_report = SchemaValidator.validate(raw_json)
    
    return validated_report.model_dump(mode="json")
"""

from .extractor import run_example

__all__ = ["run_example"]

