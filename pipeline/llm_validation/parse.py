"""Prompt building and response parsing for LLM validation."""

import re
from typing import Any, Dict, Optional

from .config import (
    CONTEXT_WINDOW,
    MAX_CONTEXT_CHARS,
    SYSTEM_MESSAGE,
    USER_TEMPLATE,
)


def _insert_span(text: str, seed_word: str) -> str:
    """Insert <span> markers around the seed word in text.

    Finds the seed word (case-insensitive) and wraps the first
    occurrence with <span>...</span>.
    """
    # Try exact match first (case-insensitive)
    pattern = re.compile(re.escape(seed_word), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        start, end = match.start(), match.end()
        return text[:start] + "<span>" + text[start:end] + "</span>" + text[end:]
    # Fallback: return text without span (shouldn't happen often)
    return text


def build_prompt(candidate: Dict[str, Any]) -> list:
    """Build chat messages for a candidate.

    Returns list of dicts [{"role": "system", ...}, {"role": "user", ...}]
    for use with tokenizer.apply_chat_template().
    """
    text = candidate.get("text", "")
    matched_sentence = candidate.get("matched_sentence", "")
    seed_word = candidate.get("seed_word", "")

    # Truncate context if too long, centered on matched sentence
    context = text
    if len(context) > MAX_CONTEXT_CHARS:
        idx = context.find(matched_sentence)
        if idx >= 0:
            start = max(0, idx - CONTEXT_WINDOW)
            end = min(len(context), idx + len(matched_sentence) + CONTEXT_WINDOW)
            context = context[start:end]
        else:
            context = context[:MAX_CONTEXT_CHARS]

    # Insert <span> around seed word
    context_with_span = _insert_span(context, seed_word)

    user_content = USER_TEMPLATE.format(context_with_span=context_with_span)

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def parse_response(text: str) -> Optional[int]:
    """Parse LLM output to extract affect score (0-3).

    Returns int 0-3, or None if parsing fails.
    """
    text = text.strip()

    # Best case: output is just a single digit
    if text in ("0", "1", "2", "3"):
        return int(text)

    # Fallback: find first digit 0-3 in the output
    match = re.search(r"[0-3]", text)
    if match:
        return int(match.group(0))

    return None
