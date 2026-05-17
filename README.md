# SignSense Backend

FastAPI backend for **SignSense**, a Gemma-powered pre-signing document risk scanner.

It accepts:

- multiple images from camera or drag-and-drop
- PDF files
- Word `.docx` files
- pasted text

It returns:

- page-aware extracted text
- plain-language summary
- risk verdict and score
- top risks with clause citations
- clean suggestions
- ask-before-signing questions
- groundedness notes when evidence is missing
- multilingual responses: English, Hindi, Telugu, Kannada, Tamil, Malayalam

## Setup

```bash
cd "SignSense Backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Google AI Studio key:

```bash
export GOOGLE_API_KEY="your_key_here"
export GEMMA_MODEL="your_gemma_4_model_name_here"
```

The default model is configurable because Google AI Studio model IDs can vary by account and release channel.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Analyze API

`POST /api/analyze`

Multipart form fields:

- `files`: one or more images/PDF/DOCX/TXT files
- `pasted_text`: optional raw text
- `input_language`: `auto`, `english`, `hindi`, `telugu`, `kannada`, `tamil`, `malayalam`
- `response_language`: same language list
- `document_type`: `auto`, `rental_pg`, `job_offer`, `loan_emi_bnpl`, `course_enrollment`, `gym_subscription`, `hostel_rules`, `hospital_consent`, `privacy_policy`, `other`

Example:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "files=@sample-rental-page-1.jpg" \
  -F "files=@sample-rental-page-2.jpg" \
  -F "response_language=english" \
  -F "document_type=rental_pg"
```

## Grounding Design

SignSense uses two Gemma calls:

1. **Extraction pass** for images and PDFs, including local-script and handwritten text where readable.
2. **Risk analysis pass** over extracted document text plus local RAG snippets from `app/data/rag_guidance.json`.

The analysis prompt requires citations for concrete risks and asks the model to return “not enough evidence” when the uploaded document does not support a claim.

## Frontend Notes

The frontend should render:

- `verdict`
- `risk_score`
- `one_sentence_summary`
- `summary`
- `top_risks`
- `clean_suggestions`
- `questions_to_ask`
- `missing_or_unclear_info`
- `groundedness_notes`
- `citations`
- `extracted_pages`

Use `/api/follow-up` for document-specific follow-up questions after the first analysis.
