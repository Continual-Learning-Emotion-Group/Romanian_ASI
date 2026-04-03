"""
LLM validation of ASI candidates via Modal + vLLM.

Runs Qwen3.5-9B on A100-80GB for maximum throughput.
Uses MASIVE-style verification prompt with 0-3 Likert scale.

All prompt building, inference, and parsing happen ON the GPU container
to eliminate network round-trips. Data is uploaded/downloaded via Modal
volumes.

Usage:
    modal run pipeline/llm_validation/modal_validate.py --max-candidates 100
    modal run pipeline/llm_validation/modal_validate.py --max-candidates 10000 --shuffle
    modal run pipeline/llm_validation/modal_validate.py --resume
    modal run pipeline/llm_validation/modal_validate.py  # all 130K
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("vllm", "huggingface-hub", "transformers", "tqdm")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("ro-asi-data", create_if_missing=True)

app = modal.App("ro-asi-llm-validation")

# ---------------------------------------------------------------------------
# Prompt building & parsing (inlined to run on GPU container)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = "Ești expert în emoții și sentimente umane."

USER_TEMPLATE = """\
Stare afectivă se referă la orice termen pe care oamenii îl folosesc pentru a descrie experiențele lor de simțire, inclusiv emoții, dispoziții și expresii figurative ale sentimentelor (de ex. „a vedea negru" ca expresie a disperării, nu a culorii). Termenul dintre <span> și </span> reflectă o stare afectivă? Răspunde doar cu un singur caracter dintre următoarele: 0, 1, 2 sau 3.

0 înseamnă Nu este o stare afectivă: termenul nu se referă la o emoție, sentiment sau stare interioară.
1 înseamnă Improbabil o stare afectivă: termenul se referă la altceva decât o emoție.
2 înseamnă Probabil o stare afectivă: termenul pare să se refere la o emoție, sentiment sau stare interioară.
3 înseamnă Categoric o stare afectivă: termenul este definitiv o emoție, sentiment sau stare interioară.

Nu explica și nu preface răspunsul.

Text: Am fost la munte weekendul trecut și am <span>încredere</span> că vom merge din nou.
Răspuns: 0

Text: Mă simt <span>fericit</span> și recunoscător pentru tot ce am primit.
Răspuns: 3

Text: Sunt <span>sigur</span> că vine mâine, am vorbit cu el la telefon.
Răspuns: 0

Text: Mi-e <span>dor</span> de casa părinților, nu am mai fost de un an.
Răspuns: 3

Text: Eu nu am <span>încredere</span> în acest produs, pare de calitate slabă.
Răspuns: 1

Text: Sunt <span>confuz</span> de tot ce se întâmplă în jurul meu, nu înțeleg nimic.
Răspuns: 2

Text: Mă simt <span>tulburată</span>, sau poate că nu neapărat asta e cuvântul potrivit, doar că nu știu eu să dau un nume la ce simt.
Răspuns: 3

Text: {context_with_span}
Răspuns:"""

MAX_CONTEXT_CHARS = 5000
CONTEXT_WINDOW = 2400


def _insert_span(text: str, seed_word: str) -> str:
    pattern = re.compile(re.escape(seed_word), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        s, e = match.start(), match.end()
        return text[:s] + "<span>" + text[s:e] + "</span>" + text[e:]
    return text


def _build_prompt(candidate: dict) -> list:
    text = candidate.get("text", "")
    seed_word = candidate.get("seed_word", "")

    # Find seed word position and truncate around it
    context = text
    if len(context) > MAX_CONTEXT_CHARS:
        # Center window on the seed word
        seed_lower = seed_word.lower()
        idx = context.lower().find(seed_lower)
        if idx >= 0:
            start = max(0, idx - CONTEXT_WINDOW)
            end = min(len(context), idx + len(seed_word) + CONTEXT_WINDOW)
            context = context[start:end]
        else:
            context = context[:MAX_CONTEXT_CHARS]

    context_with_span = _insert_span(context, seed_word)
    user_content = USER_TEMPLATE.format(context_with_span=context_with_span)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def _parse_response(text: str):
    text = text.strip()
    if text in ("0", "1", "2", "3"):
        return int(text)
    match = re.search(r"[0-3]", text)
    if match:
        return int(match.group(0))
    return None


# ---------------------------------------------------------------------------
# GPU validator — does EVERYTHING on the container
# ---------------------------------------------------------------------------

DATA_PATH = "/data"
INPUT_FILE = f"{DATA_PATH}/candidates_unified.jsonl"
OUTPUT_FILE = f"{DATA_PATH}/candidates_validated.jsonl"
CHECKPOINT_FILE = f"{DATA_PATH}/llm_validation_checkpoint.jsonl"
STATS_FILE = f"{DATA_PATH}/candidates_validated.stats.json"


@app.cls(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=4 * 60 * 60,  # 4 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        DATA_PATH: data_vol,
    },
    scaledown_window=300,
)
class Validator:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=8192,
            gpu_memory_utilization=0.92,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
        )

    @modal.method()
    def validate_all(
        self,
        max_candidates: int = 0,
        shuffle: bool = False,
        resume: bool = False,
    ) -> dict:
        """Run full validation on the GPU container. Returns stats dict."""
        import time
        from tqdm import tqdm

        # Reload volume to see latest data
        data_vol.reload()

        # Load candidates
        print(f"Loading candidates from {INPUT_FILE}...")
        candidates = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    candidates.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        print(f"  {len(candidates)} candidates loaded")

        if shuffle:
            import random
            random.seed(42)
            random.shuffle(candidates)
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
            print(f"  Limited to {len(candidates)}")

        # Checkpoint / resume
        done = {}
        if resume and Path(CHECKPOINT_FILE).exists():
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        done[rec["id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass
            print(f"  Resuming: {len(done)} already processed")
        elif not resume and Path(CHECKPOINT_FILE).exists():
            Path(CHECKPOINT_FILE).unlink()

        todo = [c for c in candidates if c["id"] not in done]
        print(f"  {len(todo)} candidates to process")

        if todo:
            CHUNK_SIZE = 20_000
            tokenizer = self.llm.get_tokenizer()
            start_time = time.time()
            total_chunks = (len(todo) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(0, len(todo), CHUNK_SIZE):
                chunk_cands = todo[chunk_idx : chunk_idx + CHUNK_SIZE]
                chunk_num = chunk_idx // CHUNK_SIZE + 1
                print(f"\n--- Chunk {chunk_num}/{total_chunks} ({len(chunk_cands)} candidates) ---")

                # Build prompts + format
                formatted = []
                for c in tqdm(chunk_cands, desc="Building prompts"):
                    msgs = _build_prompt(c)
                    formatted.append(
                        tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )

                # vLLM inference
                print(f"Running inference...")
                chunk_start = time.time()
                outputs = self.llm.generate(formatted, self.sampling_params)
                chunk_elapsed = time.time() - chunk_start
                print(f"  {len(outputs)} results in {chunk_elapsed:.1f}s "
                      f"({len(outputs)/max(chunk_elapsed,0.1):.1f} candidates/sec)")

                # Parse + checkpoint
                checkpoint_f = open(CHECKPOINT_FILE, "a", encoding="utf-8")
                for cand, output in zip(chunk_cands, outputs):
                    raw = output.outputs[0].text
                    score = _parse_response(raw)
                    result = dict(cand)
                    result["llm_affect_score"] = score
                    result["llm_raw_output"] = raw.strip()
                    result["llm_model"] = MODEL_NAME
                    done[cand["id"]] = result
                    checkpoint_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                checkpoint_f.close()
                data_vol.commit()

                total_done = chunk_idx + len(chunk_cands)
                total_elapsed = time.time() - start_time
                print(f"  Progress: {total_done}/{len(todo)} "
                      f"({total_done/max(total_elapsed,0.1):.1f} candidates/sec overall)")

            elapsed = time.time() - start_time
            print(f"\nInference complete: {len(todo)} results in {elapsed:.1f}s "
                  f"({len(todo)/max(elapsed,0.1):.1f} candidates/sec)")

        # Write final output
        print("Writing final output...")
        all_results = []
        for cand in candidates:
            if cand["id"] in done:
                all_results.append(done[cand["id"]])

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for rec in all_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Stats
        stats = {
            "total": len(all_results),
            "by_score": {},
            "by_source": {},
            "by_pattern": {},
            "parse_failures": 0,
        }

        for rec in all_results:
            score = rec.get("llm_affect_score")
            if score is None:
                stats["parse_failures"] += 1
                score_key = "null"
            else:
                score_key = str(score)
            stats["by_score"][score_key] = stats["by_score"].get(score_key, 0) + 1

            src = rec.get("source", "unknown")
            stats["by_source"].setdefault(src, {})
            stats["by_source"][src][score_key] = stats["by_source"][src].get(score_key, 0) + 1

            pat = rec.get("pattern_used", "unknown")
            stats["by_pattern"].setdefault(pat, {})
            stats["by_pattern"][pat][score_key] = stats["by_pattern"][pat].get(score_key, 0) + 1

        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # Commit volume writes
        data_vol.commit()

        # Clean up checkpoint
        if Path(CHECKPOINT_FILE).exists():
            Path(CHECKPOINT_FILE).unlink()
            data_vol.commit()

        # Print summary
        n = stats["total"]
        print(f"\n{'='*60}")
        print(f"Validation complete")
        print(f"{'='*60}")
        print(f"Total: {n}")
        print(f"Parse failures: {stats['parse_failures']}")
        print(f"\nScore distribution:")
        for score in ["0", "1", "2", "3", "null"]:
            cnt = stats["by_score"].get(score, 0)
            if cnt:
                print(f"  {score}: {cnt} ({cnt/max(n,1)*100:.1f}%)")

        print(f"\nBy source:")
        for src, scores in sorted(stats["by_source"].items()):
            total_src = sum(scores.values())
            high = scores.get("2", 0) + scores.get("3", 0)
            print(f"  {src}: {total_src} total, {high} score>=2 ({high/max(total_src,1)*100:.1f}%)")

        return stats


# ---------------------------------------------------------------------------
# Local entrypoint — uploads data, triggers GPU, downloads results
# ---------------------------------------------------------------------------

LOCAL_DATA_DIR = Path(__file__).parent.parent / "data"


@app.local_entrypoint()
def main(
    max_candidates: int = 0,
    shuffle: bool = False,
    resume: bool = False,
):
    import time

    input_file = LOCAL_DATA_DIR / "candidates_unified.jsonl"

    # Upload input data to volume
    print("Uploading input data to Modal volume...")
    start = time.time()
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(input_file), "candidates_unified.jsonl")
    print(f"  Uploaded in {time.time() - start:.1f}s")

    # Run everything on GPU
    validator = Validator()
    stats = validator.validate_all.remote(
        max_candidates=max_candidates,
        shuffle=shuffle,
        resume=resume,
    )

    # Download results
    print("\nDownloading results from Modal volume...")
    for filename in ["candidates_validated.jsonl", "candidates_validated.stats.json"]:
        local_path = LOCAL_DATA_DIR / filename
        try:
            data = b"".join(data_vol.read_file(filename))
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"  {filename} -> {local_path}")
        except Exception as e:
            print(f"  {filename}: {e}")

    print("\nDone.")
