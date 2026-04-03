"""
Modal-based LLM batch inference for ASI candidate filtering.

Runs Qwen2.5-7B-Instruct on a GPU via vLLM for maximum throughput.
Much faster than sequential API calls — processes ~2,000 candidates in ~5 minutes.

Usage:
    # Test with small batch
    modal run experiments/llm_filtering/modal_filter.py --max-candidates 20

    # Full run
    modal run experiments/llm_filtering/modal_filter.py
"""

import json
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 200

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

app = modal.App("ro-asi-llm-filter")


@app.cls(
    image=vllm_image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    scaledown_window=60,
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


@app.local_entrypoint()
def main(max_candidates: int = 0, batch_size: int = BATCH_SIZE):
    import sys
    import time

    # Add project root to path for local imports
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from experiments.llm_filtering.config import FilterConfig, PROMPT_TEMPLATE
    from experiments.llm_filtering.filter_candidates import (
        build_prompt,
        format_output,
        load_candidates,
        parse_llm_response,
        write_outputs,
    )

    config = FilterConfig()
    base = Path(project_root)
    input_path = base / "data" / "asi_candidates.jsonl"
    output_path = base / "data" / "llm_filtered_candidates.jsonl"
    full_results_path = base / "data" / "llm_filter_results.jsonl"

    # Load candidates
    candidates = load_candidates(input_path, config.allowed_sources, max_candidates)
    print(f"Loaded {len(candidates)} candidates from {', '.join(sorted(config.allowed_sources))}")

    if not candidates:
        print("No candidates found.")
        return

    # Build all prompts locally
    prompts = [build_prompt(c) for c in candidates]
    print(f"Built {len(prompts)} prompts")

    # Send to Modal in batches
    filter_cls = LLMFilter()
    all_raw_outputs: list[str] = []
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} prompts)...")
        results = filter_cls.filter_batch.remote(batch)
        all_raw_outputs.extend(results)
        elapsed = time.time() - start_time
        done = i + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        print(f"    {done}/{len(prompts)} done, {rate:.1f} candidates/sec")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(all_raw_outputs)} results in {elapsed:.1f}s "
          f"({len(all_raw_outputs)/elapsed:.1f} candidates/sec)")

    # Parse and format results locally
    results = []
    for candidate, raw in zip(candidates, all_raw_outputs):
        llm_result = parse_llm_response(raw)
        llm_result["raw_response"] = raw
        llm_result["prompt_tokens"] = 0
        llm_result["completion_tokens"] = 0
        output = format_output(candidate, llm_result, config)
        results.append(output)

    # Write outputs
    write_outputs(results, output_path, full_results_path, verbose=True)
