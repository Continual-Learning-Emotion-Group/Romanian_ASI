"""
Modal-based LLM validation of Filmot API ASI candidates.

Runs Qwen2.5-7B-Instruct on a GPU via vLLM for maximum throughput.
Takes pattern-matched candidates from filmot_api_candidates.jsonl and validates
whether each is a genuine affective state expression.

Reuses prompt/parsing logic from experiments/llm_filtering/ but preserves
YouTube-specific fields and adds checkpoint/resume support.

Usage:
    # Full run
    modal run scripts/filmot_api/llm_validate.py

    # Test with small batch
    modal run scripts/filmot_api/llm_validate.py --max-candidates 500

    # Resume after interruption
    modal run scripts/filmot_api/llm_validate.py --max-candidates 500 --resume

    # Custom batch size
    modal run scripts/filmot_api/llm_validate.py --batch-size 100
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 200

# Output paths (relative to project root, resolved in local_entrypoint)
INPUT_FILE = "data/filmot_api_candidates.jsonl"
OUTPUT_VALIDATED = "data/filmot_api_llm_validated.jsonl"
OUTPUT_RESULTS = "data/filmot_api_llm_results.jsonl"
CHECKPOINT_FILE = "data/filmot_api_llm_checkpoint.jsonl"

# YouTube-specific fields to preserve in output
YOUTUBE_FIELDS = (
    "video_id", "video_title", "channel", "views",
    "duration_seconds", "upload_date", "youtube_url",
)

# Image with vLLM installed
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("vllm==0.6.6.post1", "huggingface-hub", "transformers<4.49")
)

# Persistent volume to cache model weights across runs
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("ro-asi-filmot-llm-validate")


@app.cls(
    image=vllm_image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    scaledown_window=300,
)
class LLMFilter:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    @modal.method()
    def filter_batch(self, prompts: list[str]) -> list[str]:
        """Process a batch of prompts through vLLM. Returns raw text outputs."""
        tokenizer = self.llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


def load_filmot_candidates(
    input_path: Path,
    max_candidates: int = 0,
) -> List[Dict[str, Any]]:
    """Load filmot API candidates (no source filtering needed)."""
    candidates = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
            if max_candidates > 0 and len(candidates) >= max_candidates:
                break
    return candidates


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load previously processed results from checkpoint file."""
    results: Dict[str, Dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return results
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            results[record["id"]] = record
    return results


def format_filmot_output(
    candidate: Dict[str, Any],
    llm_result: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Format candidate + LLM result, preserving YouTube fields."""
    output = {
        "id": candidate["id"],
        "text": candidate.get("text", ""),
        "matched_sentence": candidate.get("matched_sentence", ""),
        "extraction_strategy": "llm_filtering",
        "confidence": llm_result.get("confidence", 0.0),
        "seed_word": candidate.get("seed_word", ""),
        "emotion_category": candidate.get("emotion_category", []),
        "source": "filmot_api",
        "metadata": {
            "llm_model": model_name,
            "llm_reasoning": llm_result.get("reasoning", ""),
            "llm_is_affective": llm_result.get("is_affective"),
            "original_pattern": candidate.get("pattern_used", ""),
            "original_pattern_category": candidate.get("pattern_category", ""),
        },
    }
    # Preserve YouTube-specific fields at top level
    for field in YOUTUBE_FIELDS:
        if field in candidate:
            output[field] = candidate[field]
    return output


def write_outputs(
    results: List[Dict[str, Any]],
    output_path: Path,
    full_results_path: Path,
):
    """Write filtered and full results to JSONL files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_affective = 0
    n_not_affective = 0
    n_error = 0

    # Full results
    with open(full_results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Filtered (affective only)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            is_aff = result.get("metadata", {}).get("llm_is_affective")
            if is_aff is True:
                n_affective += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            elif is_aff is False:
                n_not_affective += 1
            else:
                n_error += 1

    total = n_affective + n_not_affective + n_error
    print(f"\nResults:")
    print(f"  Total candidates: {total}")
    if total > 0:
        print(f"  Affective (kept): {n_affective} "
              f"({100 * n_affective / total:.1f}%)")
    print(f"  Not affective (filtered): {n_not_affective}")
    print(f"  Errors/unparseable: {n_error}")
    print(f"\nFiltered output: {output_path}")
    print(f"Full results: {full_results_path}")


@app.local_entrypoint()
def main(
    max_candidates: int = 0,
    batch_size: int = BATCH_SIZE,
    resume: bool = False,
):
    import sys
    import time

    # Add project root to path for local imports
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from experiments.llm_filtering.filter_candidates import (
        build_prompt,
        parse_llm_response,
    )

    base = Path(project_root)
    input_path = base / INPUT_FILE
    output_path = base / OUTPUT_VALIDATED
    full_results_path = base / OUTPUT_RESULTS
    checkpoint_path = base / CHECKPOINT_FILE

    print("Filmot API — LLM Validation (Modal + vLLM)")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {batch_size}")
    print(f"Input: {input_path}")

    # Check input exists
    if not input_path.exists():
        print(f"\nError: Input not found at {input_path}")
        print("Run filtering first:")
        print("  python -m scripts.filmot_api.filter_candidates")
        return

    # Load candidates
    candidates = load_filmot_candidates(input_path, max_candidates)
    print(f"\nLoaded {len(candidates)} filmot_api candidates")

    if not candidates:
        print("No candidates found.")
        return

    # Handle checkpoint/resume
    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint_path.exists():
        existing_results = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {len(existing_results)} already processed")
    elif not resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    # Filter out already-processed candidates
    processed_ids = set(existing_results.keys())
    to_process = [c for c in candidates if c["id"] not in processed_ids]
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining: {len(to_process)}")

    if not to_process:
        # Still write outputs from checkpoint data
        all_results = list(existing_results.values())
        write_outputs(all_results, output_path, full_results_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint cleaned up.")
        return

    # Build all prompts locally
    prompts = [build_prompt(c) for c in to_process]
    print(f"Built {len(prompts)} prompts")

    # Send to Modal in batches with checkpointing
    filter_cls = LLMFilter()
    all_results: List[Dict[str, Any]] = list(existing_results.values())
    start_time = time.time()

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_candidates = to_process[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"  Batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)...")
        raw_outputs = filter_cls.filter_batch.remote(batch_prompts)

        # Parse and format this batch
        batch_results = []
        for candidate, raw in zip(batch_candidates, raw_outputs):
            llm_result = parse_llm_response(raw)
            llm_result["raw_response"] = raw
            output = format_filmot_output(candidate, llm_result, MODEL_NAME)
            batch_results.append(output)
            all_results.append(output)

        # Checkpoint after each batch
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            for output in batch_results:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

        done = i + len(batch_prompts)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        n_affective = sum(
            1 for o in batch_results
            if o["metadata"].get("llm_is_affective") is True
        )
        print(f"    {len(processed_ids) + done}/{len(candidates)} done, "
              f"{n_affective}/{len(batch_prompts)} affective, "
              f"{rate:.1f} candidates/sec")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(to_process)} candidates in {elapsed:.1f}s "
          f"({len(to_process)/elapsed:.1f} candidates/sec)")

    # Write final outputs
    write_outputs(all_results, output_path, full_results_path)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up.")
