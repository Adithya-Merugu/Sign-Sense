from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional, Union

from app.config import settings
from app.models import AnalysisResponse, Citation, ExtractionResult, ExtractedPage, FollowUpResponse
from app.services.files import StoredUpload
from app.services.json_utils import parse_json_object
from app.services.rag import retrieve_guidance


class GemmaService:
    def __init__(self) -> None:
        self.model = settings.gemma_model
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add your Google AI Studio key to the environment.")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=settings.google_api_key)
        return self._client

    def extract_from_visual_files(
        self,
        files: list[StoredUpload],
        input_language: str,
    ) -> ExtractionResult:
        started = time.perf_counter()
        logger = logging.getLogger("signsense.gemma")
        if not files:
            return ExtractionResult(pages=[])

        logger.info("gemma.extract.upload_start files=%s", len(files))
        uploaded_files = [self.client.files.upload(file=item.path) for item in files]
        logger.info("gemma.extract.upload_done files=%s elapsed=%.2fs", len(uploaded_files), time.perf_counter() - started)
        prompt = _extraction_prompt(input_language=input_language, files=files)
        logger.info("gemma.extract.generate_start model=%s", self.model)
        response = _run_with_timeout(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=[prompt, *uploaded_files],
            ),
            timeout_seconds=settings.gemma_request_timeout_seconds,
            label="Gemma extraction",
        )
        logger.info("gemma.extract.generate_done elapsed=%.2fs", time.perf_counter() - started)
        parsed = parse_json_object(response.text or "{}")
        pages = [ExtractedPage(**page) for page in parsed.get("pages", [])]
        return ExtractionResult(
            pages=pages,
            detected_languages=parsed.get("detected_languages", []),
            warnings=parsed.get("warnings", []),
        )

    def analyze_document(
        self,
        extracted_pages: list[ExtractedPage],
        pasted_text: str,
        document_type: str,
        input_language: str,
        response_language: str,
    ) -> AnalysisResponse:
        started = time.perf_counter()
        logger = logging.getLogger("signsense.gemma")
        document_text = _compose_document_text(extracted_pages=extracted_pages, pasted_text=pasted_text)
        logger.info("gemma.analysis.compose_done chars=%s elapsed=%.2fs", len(document_text), time.perf_counter() - started)
        guidance = retrieve_guidance(document_type=document_type, document_text=document_text)
        logger.info("gemma.analysis.rag_done snippets=%s elapsed=%.2fs", len(guidance), time.perf_counter() - started)
        prompt = _analysis_prompt(
            document_text=document_text,
            document_type=document_type,
            input_language=input_language,
            response_language=response_language,
            guidance=guidance,
        )
        logger.info("gemma.analysis.generate_start model=%s prompt_chars=%s", self.model, len(prompt))
        response = _run_with_timeout(
            lambda: self.client.models.generate_content(model=self.model, contents=prompt),
            timeout_seconds=settings.gemma_request_timeout_seconds,
            label="Gemma analysis",
        )
        logger.info("gemma.analysis.generate_done elapsed=%.2fs response_chars=%s", time.perf_counter() - started, len(response.text or ""))
        parsed = parse_json_object(response.text or "{}")
        parsed["extracted_pages"] = [page.model_dump() for page in extracted_pages]
        parsed.setdefault("detected_languages", _detected_languages(extracted_pages))
        parsed.setdefault("document_type", document_type)
        parsed.setdefault("disclaimer", AnalysisResponse.model_fields["disclaimer"].default)
        return AnalysisResponse(**parsed)

    def answer_follow_up(
        self,
        question: str,
        response_language: str,
        analysis: AnalysisResponse,
    ) -> FollowUpResponse:
        prompt = _followup_prompt(question=question, response_language=response_language, analysis=analysis)
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        parsed = parse_json_object(response.text or "{}")
        return FollowUpResponse(
            answer=parsed.get("answer", "Not enough evidence in the uploaded document."),
            citations=[Citation(**item) for item in parsed.get("citations", [])],
            groundedness_notes=parsed.get("groundedness_notes", []),
        )


def _compose_document_text(extracted_pages: list[ExtractedPage], pasted_text: str) -> str:
    chunks: list[str] = []
    for page in extracted_pages:
        chunks.append(
            f"[SOURCE: {page.source_name} | PAGE: {page.page_number} | LANGUAGE: {page.language}]\n{page.text}"
        )
    if pasted_text.strip():
        chunks.append(f"[SOURCE: pasted_text | PAGE: 1 | LANGUAGE: user-provided]\n{pasted_text.strip()}")
    return "\n\n---\n\n".join(chunks).strip()


def _detected_languages(pages: list[ExtractedPage]) -> list[str]:
    languages = sorted({page.language for page in pages if page.language and page.language != "unknown"})
    return languages or ["unknown"]


def _extraction_prompt(input_language: str, files: list[StoredUpload]) -> str:
    file_list = "\n".join(f"- {item.source_name} ({item.mime_type})" for item in files)
    return f"""
You are SignSense OCR, powered by Gemma. Extract visible text from the uploaded document files.

Files:
{file_list}

Input language preference: {input_language}
Supported languages: English, Hindi, Telugu, Kannada, Tamil, Malayalam. Documents may be mixed-language and may include handwritten text.

Rules:
- Preserve page order and source filename.
- Extract handwritten additions, stamps, amounts, dates, and marginal notes when readable.
- If handwriting is unclear, include a warning and mark confidence as low.
- Do not summarize or analyze yet.
- Keep the extracted text concise but complete for legal/financial risk analysis.
- Return only valid JSON. No markdown.

JSON shape:
{{
  "detected_languages": ["english"],
  "warnings": ["string"],
  "pages": [
    {{
      "page_number": 1,
      "source_name": "filename.jpg",
      "language": "english/hindi/telugu/kannada/tamil/malayalam/mixed/unknown",
      "text": "full extracted text",
      "confidence": "low/medium/high",
      "warnings": ["unclear handwriting near the signature"]
    }}
  ]
}}
""".strip()


def _run_with_timeout(callable_obj: Any, timeout_seconds: int, label: str) -> Any:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_obj)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(
            f"{label} took longer than {timeout_seconds} seconds. Try fewer pages, a smaller file, or paste extracted text."
        ) from exc
    finally:
        if future.done():
            executor.shutdown(wait=False, cancel_futures=True)


def _analysis_prompt(
    document_text: str,
    document_type: str,
    input_language: str,
    response_language: str,
    guidance: list[dict[str, Union[str, list[str]]]],
) -> str:
    guidance_text = json.dumps(guidance, ensure_ascii=False, indent=2)
    return f"""
You are SignSense, a grounded pre-signing document risk scanner powered by Gemma.

Task:
Analyze the uploaded agreement/document and explain risks before the user signs.

Document type: {document_type}
Input language: {input_language}
Response language: {response_language}

Grounding rules:
- Use ONLY the uploaded document text and the trusted guidance snippets below.
- Do not invent clauses, amounts, dates, rights, laws, or outcomes.
- Every concrete risk must include a citation with source, page, and exact short quote from the document.
- If something is unclear or missing, put it under missing_or_unclear_info.
- Say "not enough evidence" when the document does not support an answer.
- This is not legal, medical, or financial advice. Help the user ask better questions.
- Keep the response concise. Return at most 5 top risks, 6 suggestions, and 6 questions.
- Each explanation should be 1-2 short sentences.
- Citation quotes should be short excerpts, not full paragraphs.

Trusted RAG snippets:
{guidance_text}

Uploaded document text:
{document_text}

Return only valid JSON in the response language where possible. Keep keys in English exactly as shown.
JSON shape:
{{
  "verdict": "Safe to Sign / Ask First / High Risk / Do Not Sign Yet",
  "risk_score": 0,
  "one_sentence_summary": "string",
  "document_type": "string",
  "detected_languages": ["english"],
  "summary": {{
    "plain_language_summary": "string",
    "parties_involved": ["string"],
    "key_dates": ["string"],
    "money_amounts": ["string"],
    "user_obligations": ["string"],
    "other_party_obligations": ["string"]
  }},
  "top_risks": [
    {{
      "title": "string",
      "severity": "Low / Medium / High",
      "category": "money / lock-in / refund / privacy / termination / penalty / unclear obligation / other",
      "plain_meaning": "string",
      "why_it_matters": "string",
      "suggestion": "string",
      "ask_before_signing": "string",
      "citations": [
        {{"source": "filename or pasted_text", "page": 1, "quote": "exact short quote"}}
      ]
    }}
  ],
  "clean_suggestions": ["string"],
  "questions_to_ask": ["string"],
  "missing_or_unclear_info": ["string"],
  "groundedness_notes": ["string"],
  "citations": [
    {{"source": "filename or pasted_text", "page": 1, "quote": "exact short quote"}}
  ]
}}
""".strip()


def _followup_prompt(question: str, response_language: str, analysis: AnalysisResponse) -> str:
    return f"""
You are SignSense. Answer the user's follow-up using only the prior grounded analysis and citations.

Response language: {response_language}
Question: {question}

Prior analysis JSON:
{analysis.model_dump_json(indent=2)}

Rules:
- If the answer is not supported, say not enough evidence and ask for the missing clause/page.
- Include citations when referencing document facts.
- Return only valid JSON with keys: answer, citations, groundedness_notes.
""".strip()
