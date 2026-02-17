#!/usr/bin/env python3
"""
LLM-based filtering of ASI candidates.

Takes pattern-matched ASI candidates and uses an LLM (via Featherless AI)
to validate whether each is a genuine affective state expression.

Usage:
    python -m experiments.llm_filtering.filter_candidates
    python -m experiments.llm_filtering.filter_candidates --max-candidates 100
    python -m experiments.llm_filtering.filter_candidates --resume
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .config import FilterConfig, PROMPT_TEMPLATE


def load_candidates(
    input_path: Path,
    allowed_sources: set,
    max_candidates: int = 0,
) -> List[Dict[str, Any]]:
    """Load ASI candidates, filtering to allowed sources."""
    candidates = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") not in allowed_sources:
                continue
            candidates.append(record)
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


def build_prompt(candidate: Dict[str, Any]) -> str:
    """Build LLM prompt for a candidate."""
    # Use the full text as context (truncated to keep prompt reasonable)
    full_text = candidate.get("text", "")
    matched = candidate.get("matched_sentence", "")

    # Provide surrounding context — the full text minus the matched sentence,
    # truncated to ~500 chars on each side of the match
    context = full_text
    if len(context) > 1200:
        # Try to center around the matched sentence
        idx = context.find(matched)
        if idx >= 0:
            start = max(0, idx - 500)
            end = min(len(context), idx + len(matched) + 500)
            context = context[start:end]
        else:
            context = context[:1200]

    return PROMPT_TEMPLATE.format(
        matched_sentence=matched,
        context=context,
        seed_word=candidate.get("seed_word", ""),
        pattern_used=candidate.get("pattern_used", ""),
    )


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse LLM JSON response, handling common formatting issues."""
    text = text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract fields manually
    is_affective = None
    if re.search(r'"is_affective"\s*:\s*true', text, re.IGNORECASE):
        is_affective = True
    elif re.search(r'"is_affective"\s*:\s*false', text, re.IGNORECASE):
        is_affective = False

    confidence = 0.5
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    reasoning = reasoning_match.group(1) if reasoning_match else "parse_error"

    if is_affective is not None:
        return {
            "is_affective": is_affective,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # Complete failure
    return {
        "is_affective": None,
        "confidence": 0.0,
        "reasoning": f"failed_to_parse: {text[:200]}",
    }


async def call_llm(
    client: httpx.AsyncClient,
    prompt: str,
    config: FilterConfig,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Call the Featherless AI API for a single candidate."""
    async with semaphore:
        try:
            response = await client.post(
                f"{config.api_base_url}/chat/completions",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
                timeout=config.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            result = parse_llm_response(content)
            result["raw_response"] = content
            result["prompt_tokens"] = usage.get("prompt_tokens", 0)
            result["completion_tokens"] = usage.get("completion_tokens", 0)
            return result

        except httpx.HTTPStatusError as e:
            return {
                "is_affective": None,
                "confidence": 0.0,
                "reasoning": f"http_error_{e.response.status_code}",
                "raw_response": str(e),
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        except (httpx.RequestError, KeyError, IndexError) as e:
            return {
                "is_affective": None,
                "confidence": 0.0,
                "reasoning": f"request_error: {type(e).__name__}",
                "raw_response": str(e),
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }


def format_output(
    candidate: Dict[str, Any],
    llm_result: Dict[str, Any],
    config: FilterConfig,
) -> Dict[str, Any]:
    """Format candidate + LLM result into the common output schema."""
    return {
        "id": candidate["id"],
        "text": candidate["text"],
        "matched_sentence": candidate.get("matched_sentence", ""),
        "extraction_strategy": "llm_filtering",
        "confidence": llm_result.get("confidence", 0.0),
        "seed_word": candidate.get("seed_word", ""),
        "emotion_category": candidate.get("emotion_category", []),
        "source": candidate.get("source", "unknown"),
        "metadata": {
            "llm_model": config.model,
            "llm_reasoning": llm_result.get("reasoning", ""),
            "llm_is_affective": llm_result.get("is_affective"),
            "original_pattern": candidate.get("pattern_used", ""),
            "original_pattern_category": candidate.get("pattern_category", ""),
        },
    }


async def process_candidates(
    candidates: List[Dict[str, Any]],
    config: FilterConfig,
    api_key: str,
    checkpoint_path: Path,
    existing_results: Dict[str, Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Process all candidates through the LLM."""
    semaphore = asyncio.Semaphore(config.concurrency)
    results: List[Dict[str, Any]] = list(existing_results.values())

    # Filter out already-processed candidates
    processed_ids = set(existing_results.keys())
    to_process = [c for c in candidates if c["id"] not in processed_ids]

    if verbose:
        print(f"Already processed: {len(processed_ids)}")
        print(f"Remaining: {len(to_process)}")

    if not to_process:
        return results

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()

    async with httpx.AsyncClient(headers=headers) as client:
        # Process in batches to enable checkpointing
        batch_size = config.checkpoint_interval
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start : batch_start + batch_size]

            # Build prompts and make concurrent calls
            tasks = []
            for candidate in batch:
                prompt = build_prompt(candidate)
                tasks.append(call_llm(client, prompt, config, semaphore))

            batch_results = await asyncio.gather(*tasks)

            # Process results
            batch_outputs = []
            for candidate, llm_result in zip(batch, batch_results):
                output = format_output(candidate, llm_result, config)
                output["metadata"]["prompt_tokens"] = llm_result.get(
                    "prompt_tokens", 0
                )
                output["metadata"]["completion_tokens"] = llm_result.get(
                    "completion_tokens", 0
                )
                results.append(output)
                batch_outputs.append(output)

                total_prompt_tokens += llm_result.get("prompt_tokens", 0)
                total_completion_tokens += llm_result.get("completion_tokens", 0)

            # Checkpoint: append new results
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                for output in batch_outputs:
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")

            processed_so_far = len(processed_ids) + batch_start + len(batch)
            elapsed = time.time() - start_time
            rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0

            if verbose:
                n_affective = sum(
                    1
                    for o in batch_outputs
                    if o["metadata"].get("llm_is_affective") is True
                )
                print(
                    f"  Batch {batch_start // batch_size + 1}: "
                    f"{processed_so_far}/{len(candidates)} processed, "
                    f"{n_affective}/{len(batch)} affective, "
                    f"{rate:.1f} candidates/sec"
                )

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nTokens used: {total_prompt_tokens} prompt + "
              f"{total_completion_tokens} completion = "
              f"{total_prompt_tokens + total_completion_tokens} total")
        print(f"Time: {elapsed:.1f}s")

    return results


def write_outputs(
    results: List[Dict[str, Any]],
    output_path: Path,
    full_results_path: Path,
    verbose: bool = True,
):
    """Write filtered and full results to JSONL files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_affective = 0
    n_not_affective = 0
    n_error = 0

    # Write full results (all candidates with LLM judgments)
    with open(full_results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Write filtered results (only affective candidates)
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

    if verbose:
        total = n_affective + n_not_affective + n_error
        print(f"\nResults:")
        print(f"  Total candidates: {total}")
        print(f"  Affective (kept): {n_affective} "
              f"({100 * n_affective / total:.1f}%)" if total > 0 else "")
        print(f"  Not affective (filtered): {n_not_affective}")
        print(f"  Errors/unparseable: {n_error}")
        print(f"\nFiltered output: {output_path}")
        print(f"Full results: {full_results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based filtering of ASI candidates"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to asi_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for filtered output JSONL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent API requests (default: 4)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Max candidates to process (0 = all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        print("Error: FEATHERLESS_API_KEY environment variable not set.")
        print("Get your key from https://featherless.ai and run:")
        print("  export FEATHERLESS_API_KEY='your-key-here'")
        return 1

    # Set up paths
    base_path = Path(__file__).parent.parent.parent
    input_path = args.input or base_path / "data" / "asi_candidates.jsonl"
    output_path = args.output or base_path / "data" / "llm_filtered_candidates.jsonl"
    full_results_path = output_path.with_name("llm_filter_results.jsonl")
    checkpoint_path = output_path.with_name("llm_filter_checkpoint.jsonl")

    # Check input exists
    if not input_path.exists():
        print(f"Error: Input not found at {input_path}")
        print("Run pattern extraction first:")
        print("  python -m scripts.ro_asi.extract_candidates")
        return 1

    # Build config
    config = FilterConfig()
    if args.model:
        config.model = args.model
    config.concurrency = args.concurrency

    verbose = not args.quiet

    if verbose:
        print(f"LLM Filtering Pipeline")
        print(f"=" * 50)
        print(f"Model: {config.model}")
        print(f"API: {config.api_base_url}")
        print(f"Concurrency: {config.concurrency}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Sources: {', '.join(sorted(config.allowed_sources))}")

    # Load candidates
    candidates = load_candidates(
        input_path, config.allowed_sources, args.max_candidates
    )

    if verbose:
        print(f"\nLoaded {len(candidates)} candidates from allowed sources")

    if not candidates:
        print("No candidates found for the allowed sources.")
        return 0

    # Load checkpoint if resuming
    existing_results: Dict[str, Dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        existing_results = load_checkpoint(checkpoint_path)
        if verbose:
            print(f"Resuming from checkpoint: {len(existing_results)} already processed")
    elif not args.resume and checkpoint_path.exists():
        # Fresh run — clear old checkpoint
        checkpoint_path.unlink()

    # Run LLM filtering
    if verbose:
        print(f"\nProcessing candidates...")

    results = asyncio.run(
        process_candidates(
            candidates,
            config,
            api_key,
            checkpoint_path,
            existing_results,
            verbose=verbose,
        )
    )

    # Write final outputs
    write_outputs(results, output_path, full_results_path, verbose=verbose)

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        if verbose:
            print("Checkpoint cleaned up.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
