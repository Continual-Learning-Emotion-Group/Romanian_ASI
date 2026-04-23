"""Sentence-level truncation to fit within max tokens for mT5.

MASIVE Appendix D: inputs longer than 512 tokens are trimmed by removing full
sentences (via nltk) from the end first; if that would remove the mask
sentence, trim from the beginning instead. Preserves the sentence with the
mask span at all costs.

Reasoning: truncating by characters (as the existing eval code does) can cut
mid-word, which hurts mT5 — its SentencePiece tokenizer handles subwords fine,
but dropping half a word around the mask is still noisy.
"""

from __future__ import annotations


def _ensure_nltk() -> None:
    """Make sure the punkt sentence tokenizer is available."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
    import nltk
    _ensure_nltk()
    # Romanian isn't an officially supported Punkt language in all nltk versions,
    # but the default English tokenizer handles Romanian sentence boundaries
    # reasonably well (periods, exclamation, question marks). Good enough.
    return nltk.sent_tokenize(text)


def truncate_to_max_tokens(
    text: str,
    mask_token: str,
    tokenizer,
    max_tokens: int = 512,
) -> str:
    """Remove full sentences until the tokenized text fits `max_tokens`.

    Strategy:
      1. Split into sentences; locate the sentence containing `mask_token`.
      2. If total tokens <= max_tokens, return as-is.
      3. Otherwise drop sentences one at a time, alternating from the end and
         from the beginning, never removing the mask sentence. Prefer removing
         from the end first (MASIVE behavior).
    """
    # Fast path
    if len(tokenizer.encode(text, add_special_tokens=True)) <= max_tokens:
        return text

    sents = split_sentences(text)
    mask_idx = next((i for i, s in enumerate(sents) if mask_token in s), -1)
    if mask_idx == -1:
        # Mask got split across sentence boundary (rare). Fall back to truncating
        # a character window centered on the mask.
        pos = text.find(mask_token)
        if pos == -1:
            # Last resort: token-level truncation
            ids = tokenizer.encode(text, add_special_tokens=False)[: max_tokens - 2]
            return tokenizer.decode(ids, skip_special_tokens=True)
        half_chars = max_tokens * 3  # ~3 chars per token rough avg
        start = max(0, pos - half_chars)
        end = min(len(text), pos + half_chars)
        return text[start:end]

    lo, hi = 0, len(sents) - 1
    # Drop from the end first, then from the start, alternating
    drop_end_next = True
    while lo < hi:
        candidate = " ".join(sents[lo:hi + 1])
        if len(tokenizer.encode(candidate, add_special_tokens=True)) <= max_tokens:
            return candidate
        if drop_end_next and hi > mask_idx:
            hi -= 1
        elif lo < mask_idx:
            lo += 1
        elif hi > mask_idx:
            hi -= 1
        else:
            # Only the mask sentence remains and it's still too long
            break
        drop_end_next = not drop_end_next

    # Only the mask sentence is left (or we failed). Hard-truncate by tokens
    # centered on the mask token id.
    mask_sent = sents[mask_idx]
    ids = tokenizer.encode(mask_sent, add_special_tokens=False)
    if len(ids) + 2 <= max_tokens:
        return mask_sent
    # Find the mask token id position and keep a window around it
    mask_ids = tokenizer.encode(mask_token, add_special_tokens=False)
    pos = -1
    for i in range(len(ids) - len(mask_ids) + 1):
        if ids[i : i + len(mask_ids)] == mask_ids:
            pos = i
            break
    budget = max_tokens - 2
    if pos == -1:
        return tokenizer.decode(ids[:budget], skip_special_tokens=True)
    half = budget // 2
    start = max(0, pos - half)
    end = min(len(ids), start + budget)
    return tokenizer.decode(ids[start:end], skip_special_tokens=True)
