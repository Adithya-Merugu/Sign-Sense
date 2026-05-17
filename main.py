from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.models import (
    AnalysisResponse,
    DocumentType,
    ExtractionResult,
    ExtractedPage,
    FollowUpRequest,
    FollowUpResponse,
    SupportedLanguage,
)
from app.services.files import StoredUpload, cleanup_uploads, store_uploads
from app.services.gemma import GemmaService
from app.services.local_extract import extract_text_locally


app = FastAPI(title="SignSense Backend", version="0.1.0")
gemma = GemmaService()
logger = logging.getLogger("signsense")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in settings.allowed_origins else list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": settings.gemma_model,
        "timeout_seconds": str(settings.gemma_request_timeout_seconds),
        "api_key_configured": "yes" if settings.google_api_key else "no",
    }


@app.get("/debug/cors")
def debug_cors() -> dict[str, object]:
    return {
        "allowed_origins": list(settings.allowed_origins),
        "allow_credentials": False,
    }


@app.post("/api/extract", response_model=ExtractionResult)
async def extract_document(
    files: Optional[list[UploadFile]] = File(default=None),
    pasted_text: str = Form(default=""),
    input_language: SupportedLanguage = Form(default=SupportedLanguage.auto),
) -> ExtractionResult:
    stored: list[StoredUpload] = []
    started = time.perf_counter()
    try:
        logger.info("extract.request started files=%s pasted_chars=%s input_language=%s", len(files or []), len(pasted_text), input_language.value)
        stored = await store_uploads(files, max_upload_mb=settings.max_upload_mb)
        logger.info("extract.uploads_stored count=%s elapsed=%.2fs", len(stored), time.perf_counter() - started)
        pages = await _extract_pages(stored=stored, input_language=input_language.value)
        logger.info("extract.pages_ready count=%s elapsed=%.2fs", len(pages), time.perf_counter() - started)
        if pasted_text.strip():
            pages.append(
                ExtractedPage(
                    page_number=1,
                    source_name="pasted_text",
                    language=input_language.value,
                    text=pasted_text.strip(),
                    confidence="high",
                )
            )
        return ExtractionResult(pages=pages, detected_languages=_detected_languages(pages))
    except ValueError as exc:
        logger.exception("extract.request value_error")
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("extract.request runtime_error")
        status_code = 504 if "took longer" in str(exc) else 500
        raise HTTPException(status_code=status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("extract.request unexpected_error")
        raise HTTPException(status_code=500, detail=f"Unexpected extraction error: {exc}")
    finally:
        logger.info("extract.request finished elapsed=%.2fs", time.perf_counter() - started)
        cleanup_uploads(stored)


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_document(
    files: Optional[list[UploadFile]] = File(default=None),
    pasted_text: str = Form(default=""),
    input_language: SupportedLanguage = Form(default=SupportedLanguage.auto),
    response_language: SupportedLanguage = Form(default=SupportedLanguage.english),
    document_type: DocumentType = Form(default=DocumentType.auto),
) -> AnalysisResponse:
    stored: list[StoredUpload] = []
    started = time.perf_counter()
    try:
        logger.info(
            "analyze.request started files=%s pasted_chars=%s input_language=%s response_language=%s document_type=%s",
            len(files or []),
            len(pasted_text),
            input_language.value,
            response_language.value,
            document_type.value,
        )
        stored = await store_uploads(files, max_upload_mb=settings.max_upload_mb)
        logger.info("analyze.uploads_stored count=%s elapsed=%.2fs", len(stored), time.perf_counter() - started)
        pages = await _extract_pages(stored=stored, input_language=input_language.value)
        logger.info(
            "analyze.extraction_done pages=%s chars=%s elapsed=%.2fs",
            len(pages),
            sum(len(page.text or "") for page in pages) + len(pasted_text),
            time.perf_counter() - started,
        )

        if not pages and not pasted_text.strip():
            raise HTTPException(status_code=400, detail="Upload at least one file or paste document text.")

        logger.info("analyze.gemma_analysis_start elapsed=%.2fs", time.perf_counter() - started)
        result = await run_in_threadpool(
            gemma.analyze_document,
            extracted_pages=pages,
            pasted_text=pasted_text,
            document_type=document_type.value,
            input_language=input_language.value,
            response_language=response_language.value,
        )
        logger.info(
            "analyze.gemma_analysis_done verdict=%s risks=%s elapsed=%.2fs",
            result.verdict,
            len(result.top_risks),
            time.perf_counter() - started,
        )
        return result
    except ValueError as exc:
        logger.exception("analyze.request value_error")
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("analyze.request runtime_error")
        status_code = 504 if "took longer" in str(exc) else 500
        raise HTTPException(status_code=status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("analyze.request unexpected_error")
        raise HTTPException(status_code=500, detail=f"Unexpected analysis error: {exc}")
    finally:
        logger.info("analyze.request finished elapsed=%.2fs", time.perf_counter() - started)
        cleanup_uploads(stored)


@app.post("/api/follow-up", response_model=FollowUpResponse)
def follow_up(payload: FollowUpRequest) -> FollowUpResponse:
    started = time.perf_counter()
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        logger.info("follow_up.request started question_chars=%s language=%s", len(payload.question), payload.response_language.value)
        return gemma.answer_follow_up(
            question=payload.question.strip(),
            response_language=payload.response_language.value,
            analysis=payload.analysis,
        )
    except RuntimeError as exc:
        logger.exception("follow_up.request runtime_error")
        status_code = 504 if "took longer" in str(exc) else 500
        raise HTTPException(status_code=status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("follow_up.request unexpected_error")
        raise HTTPException(status_code=500, detail=f"Unexpected follow-up error: {exc}")
    finally:
        logger.info("follow_up.request finished elapsed=%.2fs", time.perf_counter() - started)


async def _extract_pages(stored: list[StoredUpload], input_language: str) -> list[ExtractedPage]:
    started = time.perf_counter()
    pages: list[ExtractedPage] = []
    visual_or_scanned: list[StoredUpload] = []

    for upload in stored:
        file_started = time.perf_counter()
        if upload.mime_type.startswith("image/"):
            visual_or_scanned.append(upload)
            logger.info("extract.file queued_for_gemma_vision name=%s mime=%s", upload.source_name, upload.mime_type)
            continue

        local_pages = extract_text_locally(upload)
        if local_pages:
            pages.extend(local_pages)
            logger.info(
                "extract.file local_success name=%s mime=%s pages=%s chars=%s elapsed=%.2fs",
                upload.source_name,
                upload.mime_type,
                len(local_pages),
                sum(len(page.text or "") for page in local_pages),
                time.perf_counter() - file_started,
            )
        elif upload.mime_type == "application/pdf":
            visual_or_scanned.append(upload)
            logger.info("extract.file local_empty_queued_for_gemma name=%s mime=%s elapsed=%.2fs", upload.source_name, upload.mime_type, time.perf_counter() - file_started)
        else:
            logger.info("extract.file no_text name=%s mime=%s elapsed=%.2fs", upload.source_name, upload.mime_type, time.perf_counter() - file_started)

    if visual_or_scanned:
        logger.info("extract.gemma_vision_start files=%s elapsed=%.2fs", len(visual_or_scanned), time.perf_counter() - started)
        gemma_result = await run_in_threadpool(
            gemma.extract_from_visual_files,
            visual_or_scanned,
            input_language,
        )
        pages.extend(gemma_result.pages)
        logger.info(
            "extract.gemma_vision_done pages=%s warnings=%s elapsed=%.2fs",
            len(gemma_result.pages),
            len(gemma_result.warnings),
            time.perf_counter() - started,
        )

    logger.info("extract.complete pages=%s elapsed=%.2fs", len(pages), time.perf_counter() - started)
    return pages


def _detected_languages(pages: list[ExtractedPage]) -> list[str]:
    languages = sorted({page.language for page in pages if page.language and page.language != "unknown"})
    return languages or ["unknown"]
