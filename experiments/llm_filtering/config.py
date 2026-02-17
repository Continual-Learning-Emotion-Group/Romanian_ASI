"""Configuration for LLM-based ASI candidate filtering."""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class FilterConfig:
    """Configuration for the LLM filtering pipeline."""

    # Featherless AI API settings
    api_base_url: str = "https://api.featherless.ai/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 256

    # Concurrency / rate limiting
    concurrency: int = 4
    request_timeout: float = 30.0

    # Checkpoint settings
    checkpoint_interval: int = 100  # save progress every N candidates

    # Source filtering — only process these sources
    allowed_sources: Set[str] = field(
        default_factory=lambda: {"reddit_roap", "poprero"}
    )


# Romanian prompt template for affective state validation
PROMPT_TEMPLATE = """\
Ești un lingvist expert în limba română. Analizează propoziția de mai jos și decide dacă exprimă o STARE AFECTIVĂ a vorbitorului (emoție, sentiment, dispoziție).

Propoziție: "{matched_sentence}"
Context: "{context}"
Cuvântul detectat: "{seed_word}"
Tiparul folosit: "{pattern_used}"

Răspunde DOAR cu un JSON valid, fără alte explicații:
{{"is_affective": true/false, "confidence": 0.0-1.0, "reasoning": "explicație scurtă"}}

Reguli:
- TRUE: propoziția exprimă ce simte/resimte vorbitorul (emoții, sentimente, stări fizice legate de emoții)
  Exemple: "sunt trist", "mă simt fericit", "mi-e frică", "am fost dezamăgit"
- FALSE: propoziția folosește cuvântul în sens non-afectiv
  Exemple: "sunt sigur că..." (certitudine, nu emoție), "sunt student" (identitate), "am fost acolo" (locație)
- Fii atent la sensul din context, nu doar la cuvânt"""
