"""Centralised evaluation metrics for ASI masked-word prediction.

Metrics follow MASIVE Section 4.2:
  - Top-k accuracy (k=1,3,5)
  - Top-k similarity (k=1,3,5) via contextual embeddings
  - MRR (Mean Reciprocal Rank)
"""

import re
from collections import defaultdict

import numpy as np
import torch

from pipeline.utils.text_utils import normalize_text
from pipeline.utils.stoplists import infer_gender


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_prediction(pred: str) -> str:
    """Lowercase + remove diacritics + strip punctuation/whitespace."""
    normed = normalize_text(pred)
    normed = _PUNCT_RE.sub("", normed)
    return normed.strip()


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(gold_normalized: str, predictions_normalized: list[str], k: int) -> float:
    """1.0 if gold is in top-k predictions (exact match), else 0.0."""
    return 1.0 if gold_normalized in predictions_normalized[:k] else 0.0


def mrr(gold_normalized: str, predictions_normalized: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first exact match, or 0.0."""
    for i, pred in enumerate(predictions_normalized):
        if pred == gold_normalized:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Contextual similarity scorer
# ---------------------------------------------------------------------------

class ContextualSimilarityScorer:
    """Compute top-k similarity using contextual embeddings.

    Follows MASIVE: embed each word in context using multilingual BERT,
    compute cosine similarity between prediction and gold embeddings.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-multilingual-cased",
        device: str | None = None,
        max_context_tokens: int = 128,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_context_tokens = max_context_tokens

    @torch.no_grad()
    def _embed_word_in_context(self, word: str, masked_text: str) -> np.ndarray:
        """Embed *word* by substituting it at [MASK] in *masked_text*.

        Returns the averaged hidden-state vector for the word's tokens.
        """
        text = masked_text.replace("[MASK]", word, 1)
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_context_tokens,
        ).to(self.device)

        hidden = self.model(**enc).last_hidden_state[0]  # (seq_len, dim)

        # Find which tokens correspond to *word*
        word_enc = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        input_ids = enc["input_ids"][0].tolist()

        # Sliding window search for the word tokens
        for start in range(len(input_ids) - len(word_enc) + 1):
            if input_ids[start : start + len(word_enc)] == word_enc:
                vecs = hidden[start : start + len(word_enc)]
                return vecs.mean(dim=0).cpu().numpy()

        # Fallback: average all non-special tokens
        return hidden[1:-1].mean(dim=0).cpu().numpy()

    def top_k_similarity(
        self,
        gold_word: str,
        predictions: list[str],
        masked_text: str,
        k: int,
    ) -> float:
        """Max cosine similarity between gold and top-k predictions, in context."""
        gold_vec = self._embed_word_in_context(gold_word, masked_text)
        gold_norm = np.linalg.norm(gold_vec)
        if gold_norm < 1e-9:
            return 0.0

        best_sim = -1.0
        for pred in predictions[:k]:
            pred_vec = self._embed_word_in_context(pred, masked_text)
            pred_norm = np.linalg.norm(pred_vec)
            if pred_norm < 1e-9:
                continue
            sim = float(np.dot(gold_vec, pred_vec) / (gold_norm * pred_norm))
            if sim > best_sim:
                best_sim = sim

        return max(best_sim, 0.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: list[dict],
    scorer: ContextualSimilarityScorer | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Aggregate metrics over all samples.

    Each result dict must have:
      - gold_normalized: str
      - predictions_normalized: list[str]
      - masked_text: str  (only needed if scorer is provided)
      - seed_word: str    (only needed if scorer is provided)
    """
    acc = {k: [] for k in ks}
    sim = {k: [] for k in ks}
    mrr_scores = []

    for r in results:
        gold = r["gold_normalized"]
        preds = r["predictions_normalized"]

        for k in ks:
            acc[k].append(top_k_accuracy(gold, preds, k))

        mrr_scores.append(mrr(gold, preds))

        if scorer is not None:
            for k in ks:
                s = scorer.top_k_similarity(
                    r["seed_word"], r["predictions_raw"][:k], r["masked_text"], k,
                )
                sim[k].append(s)

    metrics: dict = {"n_samples": len(results)}
    for k in ks:
        metrics[f"acc@{k}"] = float(np.mean(acc[k])) if acc[k] else 0.0
    metrics["mrr"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    if scorer is not None:
        for k in ks:
            metrics[f"sim@{k}"] = float(np.mean(sim[k])) if sim[k] else 0.0

    return metrics


def compute_metrics_by_group(
    results: list[dict],
    group_key: str,
    scorer: ContextualSimilarityScorer | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, dict]:
    """Compute metrics per group (source, pattern_category, gender, etc.).

    If group_key == "gender", infer gender from seed_word via stoplists.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        if group_key == "gender":
            g = infer_gender(r.get("seed_word", "")) or "unknown"
        else:
            val = r.get(group_key, "unknown")
            g = str(val) if not isinstance(val, list) else ",".join(val)
        groups[g].append(r)

    return {
        g: compute_all_metrics(group_results, scorer=scorer, ks=ks)
        for g, group_results in sorted(groups.items())
    }
