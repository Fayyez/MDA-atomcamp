import sys
import os
from pathlib import Path

# Add project root to sys.path so 'utils' can be found when script is run directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from typing import Union
from utils.report_extractor.preprocessing.image_preprocessor import ImagePreprocessor
from utils.report_extractor.extraction.vlm_engine import VLMEngine
from utils.report_extractor.validation.schema_validator import SchemaValidator, MedicalReport


def extract_from_image(image: Union[Image.Image, str]) -> dict:
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
        
    preprocessor = ImagePreprocessor()
    clean_img = preprocessor.preprocess(img)

    vlm = VLMEngine()
    raw_json = vlm.extract(clean_img, MedicalReport)

    validated_report = SchemaValidator.validate(raw_json)
    return validated_report.model_dump(mode="json")


def run_example(image_path: str = None) -> dict:
    if image_path is None:
        image_path = str(Path(__file__).parent / "input_reports/report.png")
    return extract_from_image(image_path)


if __name__ == "__main__":
    report = run_example()
    print(report)
