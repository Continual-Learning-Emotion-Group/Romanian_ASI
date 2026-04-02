"""
Romanian text utilities for the ASI pipeline.

Diacritics normalization, sentence splitting, and context extraction.
"""

import re
from typing import List, Tuple

# Romanian diacritic mappings for normalization
DIACRITIC_MAP = {
    'ă': 'a', 'â': 'a', 'î': 'i',
    'ș': 's', 'ț': 't',
    'Ă': 'A', 'Â': 'A', 'Î': 'I',
    'Ș': 'S', 'Ț': 'T',
    # Common OCR/typing errors (cedilla variants)
    'ş': 's', 'ţ': 't',
    'Ş': 'S', 'Ţ': 'T',
}


def remove_diacritics(text: str) -> str:
    """Remove Romanian diacritics from text."""
    result = []
    for char in text:
        result.append(DIACRITIC_MAP.get(char, char))
    return ''.join(result)


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase and remove diacritics."""
    return remove_diacritics(text.lower())


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into sentences with positions.

    Returns list of (start_pos, end_pos, sentence_text) tuples.
    """
    sentence_pattern = r'[.!?]+(?:\s+|$)'

    sentences = []
    last_end = 0

    for match in re.finditer(sentence_pattern, text):
        end_pos = match.end()
        sentence = text[last_end:end_pos].strip()
        if sentence and len(sentence) > 10:
            sentences.append((last_end, end_pos, sentence))
        last_end = end_pos

    # Remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining and len(remaining) > 10:
            sentences.append((last_end, len(text), remaining))

    return sentences


def extract_context_window(
    text: str,
    match_start: int,
    match_end: int,
    num_sentences: int = 2,
    max_length: int = 1000,
) -> Tuple[str, str, str]:
    """
    Extract context window around a match using sentence boundaries.

    Returns (context_before, matched_sentence, context_after).
    """
    sentences = split_into_sentences(text)

    if not sentences:
        start = max(0, match_start - 200)
        end = min(len(text), match_end + 200)
        return ("", text[start:end].strip(), "")

    # Find sentence containing the match
    match_sentence_idx = -1
    for i, (start, end, sent) in enumerate(sentences):
        if start <= match_start < end:
            match_sentence_idx = i
            break

    if match_sentence_idx == -1:
        start = max(0, match_start - 200)
        end = min(len(text), match_end + 200)
        return ("", text[start:end].strip(), "")

    matched_sentence = sentences[match_sentence_idx][2]

    # Context before
    before_sentences = []
    for i in range(max(0, match_sentence_idx - num_sentences), match_sentence_idx):
        before_sentences.append(sentences[i][2])
    context_before = " ".join(before_sentences)

    # Context after
    after_sentences = []
    for i in range(match_sentence_idx + 1, min(len(sentences), match_sentence_idx + num_sentences + 1)):
        after_sentences.append(sentences[i][2])
    context_after = " ".join(after_sentences)

    # Trim if too long
    total_len = len(context_before) + len(matched_sentence) + len(context_after)
    if total_len > max_length:
        available = max_length - len(matched_sentence)
        half = available // 2
        if len(context_before) > half:
            context_before = "..." + context_before[-(half - 3):]
        if len(context_after) > half:
            context_after = context_after[:half - 3] + "..."

    return (context_before, matched_sentence, context_after)
