from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SupportedLanguage(str, Enum):
    auto = "auto"
    english = "english"
    hindi = "hindi"
    telugu = "telugu"
    kannada = "kannada"
    tamil = "tamil"
    malayalam = "malayalam"


class DocumentType(str, Enum):
    auto = "auto"
    rental_pg = "rental_pg"
    job_offer = "job_offer"
    loan_emi_bnpl = "loan_emi_bnpl"
    course_enrollment = "course_enrollment"
    gym_subscription = "gym_subscription"
    hostel_rules = "hostel_rules"
    hospital_consent = "hospital_consent"
    privacy_policy = "privacy_policy"
    other = "other"


class ExtractedPage(BaseModel):
    page_number: int
    source_name: str
    language: str = "unknown"
    text: str
    confidence: str = Field(default="medium", description="low, medium, or high")
    warnings: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    pages: list[ExtractedPage]
    detected_languages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    quote: str


class RiskItem(BaseModel):
    title: str
    severity: str
    category: str
    plain_meaning: str
    why_it_matters: str
    suggestion: str
    ask_before_signing: str
    citations: list[Citation] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    verdict: str
    risk_score: int = Field(ge=0, le=100)
    one_sentence_summary: str
    document_type: str
    detected_languages: list[str]
    summary: dict[str, Any]
    top_risks: list[RiskItem]
    clean_suggestions: list[str]
    questions_to_ask: list[str]
    missing_or_unclear_info: list[str]
    groundedness_notes: list[str]
    citations: list[Citation]
    extracted_pages: list[ExtractedPage]
    disclaimer: str = (
        "SignSense is not legal, medical, or financial advice. It highlights document risks "
        "and helps you ask better questions before signing."
    )


class FollowUpRequest(BaseModel):
    question: str
    response_language: SupportedLanguage = SupportedLanguage.english
    analysis: AnalysisResponse


class FollowUpResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    groundedness_notes: list[str] = Field(default_factory=list)
