"""
LLM validation of ASI candidates via Modal + vLLM.

Runs Qwen3.5-9B on A100-80GB for maximum throughput.
Uses MASIVE-style verification prompt with 0-3 Likert scale.

Usage:
    modal run pipeline/llm_validation/modal_validate.py --max-candidates 100
    modal run pipeline/llm_validation/modal_validate.py --max-candidates 10000
    modal run pipeline/llm_validation/modal_validate.py --resume
    modal run pipeline/llm_validation/modal_validate.py --batch-size 1000 --max-candidates 50000
"""

import json
from collections import defaultdict
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("vllm", "huggingface-hub", "transformers")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("ro-asi-llm-validation")


@app.cls(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    scaledown_window=1800,  # 30 min — keep container warm between runs
)
class Validator:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=4096,
            gpu_memory_utilization=0.92,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4,
        )

    @modal.method()
    def validate_batch(self, prompts: list[str]) -> list[str]:
        """Process a batch of chat-formatted prompts through vLLM."""
        tokenizer = self.llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in prompts
        ]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "candidates_unified.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "candidates_validated.jsonl"
CHECKPOINT_PATH = DATA_DIR / "llm_validation_checkpoint.jsonl"


def _load_jsonl(path: Path) -> list:
    records = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  ({bad} bad lines skipped)")
    return records


def _load_checkpoint() -> dict:
    """Load checkpoint: returns dict of id -> result record."""
    if not CHECKPOINT_PATH.exists():
        return {}
    results = {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                results[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return results


def _append_checkpoint(records: list):
    """Append records to checkpoint file."""
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.local_entrypoint()
def main(
    max_candidates: int = 0,
    batch_size: int = 500,
    resume: bool = False,
    shuffle: bool = False,
    input_path: str = "",
    output_path: str = "",
):
    import sys
    import time

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from pipeline.llm_validation.parse import build_prompt, parse_response
    from pipeline.llm_validation.config import MODEL_NAME as model_name

    inp = Path(input_path) if input_path else DEFAULT_INPUT
    out = Path(output_path) if output_path else DEFAULT_OUTPUT

    # Load candidates
    print(f"Loading candidates from {inp.name}...")
    candidates = _load_jsonl(inp)
    if shuffle:
        import random
        random.seed(42)
        random.shuffle(candidates)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    print(f"  {len(candidates)} candidates loaded{' (shuffled)' if shuffle else ''}")

    if not candidates:
        print("No candidates found.")
        return

    # Checkpoint / resume
    done = {}
    if resume and CHECKPOINT_PATH.exists():
        done = _load_checkpoint()
        print(f"  Resuming: {len(done)} already processed")
    elif not resume and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # Filter out already-processed
    todo = [c for c in candidates if c["id"] not in done]
    print(f"  {len(todo)} candidates to process")

    if not todo:
        print("Nothing to process — writing outputs from checkpoint.")
    else:
        # Build all prompts locally
        prompts = [build_prompt(c) for c in todo]
        print(f"Built {len(prompts)} prompts")

        # Send to Modal in batches
        validator = Validator()
        start_time = time.time()
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_candidates = todo[i : i + batch_size]
            batch_num = i // batch_size + 1

            print(f"  Batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)...")
            raw_outputs = validator.validate_batch.remote(batch_prompts)

            # Parse and build result records
            batch_results = []
            for cand, raw in zip(batch_candidates, raw_outputs):
                score = parse_response(raw)
                result = dict(cand)
                result["llm_affect_score"] = score
                result["llm_raw_output"] = raw.strip()
                result["llm_model"] = model_name
                batch_results.append(result)
                done[cand["id"]] = result

            # Checkpoint
            _append_checkpoint(batch_results)

            elapsed = time.time() - start_time
            processed = i + len(batch_prompts)
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    {processed}/{len(todo)} done, {rate:.1f} candidates/sec")

        elapsed = time.time() - start_time
        print(f"\nInference complete: {len(todo)} results in {elapsed:.1f}s "
              f"({len(todo)/max(elapsed,0.1):.1f} candidates/sec)")

    # Write final output: all candidates (done dict covers everything)
    all_results = []
    for cand in candidates:
        if cand["id"] in done:
            all_results.append(done[cand["id"]])
        # else: skip candidates not yet processed (shouldn't happen unless max_candidates changed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "total": len(all_results),
        "by_score": defaultdict(int),
        "by_source": defaultdict(lambda: defaultdict(int)),
        "by_pattern": defaultdict(lambda: defaultdict(int)),
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

    # Convert defaultdicts for JSON serialization
    stats["by_score"] = dict(stats["by_score"])
    stats["by_source"] = {k: dict(v) for k, v in stats["by_source"].items()}
    stats["by_pattern"] = {k: dict(v) for k, v in stats["by_pattern"].items()}

    stats_path = out.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

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

    print(f"\nOutput: {out}")

    # Clean up checkpoint on success
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up.")
