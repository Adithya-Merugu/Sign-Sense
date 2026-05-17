from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Union


GUIDANCE_PATH = Path(__file__).resolve().parents[1] / "data" / "rag_guidance.json"


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w₹$%.-]+", text) if len(token) > 2}


@lru_cache(maxsize=1)
def load_guidance() -> list[dict[str, Union[str, list[str]]]]:
    with GUIDANCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def retrieve_guidance(document_type: str, document_text: str, limit: int = 5) -> list[dict[str, Union[str, list[str]]]]:
    query_tokens = _tokens(f"{document_type} {document_text[:8000]}")
    scored: list[tuple[int, dict[str, Union[str, list[str]]]]] = []

    for item in load_guidance():
        domains = item.get("domains", [])
        domain_score = 6 if document_type in domains or "all" in domains else 0
        item_tokens = _tokens(f"{item.get('title', '')} {item.get('content', '')}")
        overlap = len(query_tokens & item_tokens)
        scored.append((domain_score + overlap, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0]
