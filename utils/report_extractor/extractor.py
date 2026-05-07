from PIL import Image
from .preprocessing.image_preprocessor import ImagePreprocessor
from .extraction.vlm_engine import VLMEngine
from .validation.schema_validator import SchemaValidator, MedicalReport


def run_example(image_path: str = "input_reports/report2.jpg") -> dict:
    img = Image.open(image_path).convert("RGB")
    preprocessor = ImagePreprocessor()
    clean_img = preprocessor.preprocess(img)

    vlm = VLMEngine()
    raw_json = vlm.extract(clean_img, MedicalReport)

    validated_report = SchemaValidator.validate(raw_json)
    return validated_report.model_dump(mode="json")


if __name__ == "__main__":
    report = run_example()
    print(report)
