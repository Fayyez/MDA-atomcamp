# midas_extraction/validation/schema_validator.py

from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from datetime import date
from instructor import Instructor  # used later to enforce constraints in prompt


class LabTest(BaseModel):
    test_name: str = Field(..., description="Name of the lab test")
    value: str = Field(..., description="Value of the test (string, may be numeric or descriptive)")
    unit: Optional[str] = Field(default=None, description="Unit if applicable")
    reference_range: Optional[str] = Field(default=None, description="Reference range if present")
    abnormal_flag: Optional[bool] = Field(default=None, description="True if flagged abnormal")


class PatientInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Patient full name")
    age: Optional[int] = Field(default=None, description="Patient age in years")
    gender: Optional[str] = Field(default=None, description="Patient gender")
    report_date: Optional[date] = Field(default=None, description="Report issue date")


class MedicalReport(BaseModel):
    patient: Optional[PatientInfo] = Field(default=None)
    lab_tests: List[LabTest] = Field(default_factory=list)
    diagnosis: Optional[List[str]] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class SchemaValidator:
    """
    Wraps Pydantic validation for downstream use.
    """

    @staticmethod
    def validate(report_dict: dict) -> MedicalReport:
        try:
            return MedicalReport.model_validate(report_dict)
        except ValidationError as e:
            # Optionally: log or flag errors
            raise ValueError(f"Schema validation failed: {e}")