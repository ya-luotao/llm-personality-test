#!/usr/bin/env python3
"""Run MBTI tests on multiple models."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.mcp_client import OpenMBTIClient
from src.llm_client import LLMClient
from src.test_runner import MBTITestRunner
from src.models import TestSession, TestResult, QuestionAnswer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MBTI tests on multiple LLMs"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        required=True,
        help="List of OpenRouter model IDs",
    )
    parser.add_argument(
        "--run-id",
        "-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run ID for this batch (default: timestamp)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=5,
        help="Number of test runs per model (default: 5)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=3,
        help="Number of parallel runs per model (default: 3)",
    )
    parser.add_argument(
        "--continue",
        "-c",
        dest="continue_run",
        action="store_true",
        help="Continue from existing results if available",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        choices=[4, 5],
        default=5,
        help="Answer scale: 4 (forced choice, no neutral) or 5 (with neutral) (default: 5)",
    )
    return parser.parse_args()


def get_output_dir(run_id: str) -> Path:
    output_dir = Path("results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_model_output_path(output_dir: Path, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return output_dir / f"{safe_model}.json"


def load_existing_session(filepath: Path) -> TestSession | None:
    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = TestSession(
            model=data["model"],
            total_runs=data["total_runs"],
            timestamp=data["timestamp"],
        )

        for run_data in data.get("runs", []):
            questions = [
                QuestionAnswer(
                    id=q["id"],
                    text=q["text"],
                    options=q["options"],
                    llm_choice=q["llm_choice"],
                    llm_raw_response=q["llm_raw_response"],
                )
                for q in run_data.get("questions", [])
            ]
            result = TestResult(
                model_name=session.model,
                run_id=run_data["run_id"],
                timestamp=run_data["timestamp"],
                questions=questions,
                mbti_type=run_data["mbti_type"],
                dimension_scores=run_data["dimension_scores"],
                raw_response=run_data.get("raw_response", {}),
            )
            session.runs.append(result)

        return session
    except Exception as e:
        print(f"  Warning: Could not load existing session: {e}", file=sys.stderr)
        return None


def save_session(session: TestSession, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)


async def run_single_test_with_output(
    runner: MBTITestRunner,
    run_id: int,
    total_runs: int,
    model: str,
    session: TestSession,
    output_path: Path,
    lock: asyncio.Lock,
) -> TestResult | None:
    short_model = model.split("/")[-1]

    def on_progress(rid: int, completed: int, total: int):
        progress_str = f"\r  [{short_model}] Run {rid}: {completed}/{total}"
        print(progress_str, end="", flush=True)

    try:
        result = await runner.run_single_test(run_id=run_id, on_progress=on_progress)

        async with lock:
            session.runs.append(result)
            save_session(session, output_path)

            runs_completed = len(session.runs)
            choices = " ".join(str(q.llm_choice) for q in result.questions)
            print(f"\r  [{short_model}] Run {run_id}: {result.mbti_type} | {choices} ({runs_completed}/{total_runs})")

        return result
    except Exception as e:
        async with lock:
            print(f"\r  [{short_model}] Run {run_id}: Error - {e}", file=sys.stderr)
        return None


async def run_model_tests(
    model: str,
    num_runs: int,
    api_key: str,
    output_path: Path,
    parallel: int,
    mcp_client: OpenMBTIClient,
    temperature: float = 0.7,
    scale: int = 5,
    existing_session: TestSession | None = None,
) -> TestSession:
    if existing_session:
        session = existing_session
        session.total_runs = num_runs
        completed_run_ids = {r.run_id for r in session.runs}
        remaining_run_ids = [i for i in range(1, num_runs + 1) if i not in completed_run_ids]
        if remaining_run_ids:
            print(f"  Resuming: {len(session.runs)} done, {len(remaining_run_ids)} remaining")
    else:
        session = TestSession(
            model=model,
            total_runs=num_runs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        remaining_run_ids = list(range(1, num_runs + 1))

    if not remaining_run_ids:
        print(f"  Already completed all {num_runs} runs")
        return session

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(parallel)

    async def bounded_run(run_id: int):
        async with semaphore:
            llm_client = LLMClient(model=model, api_key=api_key, temperature=temperature)
            runner = MBTITestRunner(mcp_client, llm_client, scale=scale)
            return await run_single_test_with_output(
                runner, run_id, num_runs, model, session, output_path, lock
            )

    tasks = [bounded_run(run_id) for run_id in remaining_run_ids]
    await asyncio.gather(*tasks)

    return session


async def run_all_models(
    models: list[str],
    num_runs: int,
    api_key: str,
    output_dir: Path,
    parallel: int,
    continue_run: bool,
    temperature: float = 0.7,
    scale: int = 5,
) -> dict[str, TestSession]:
    results: dict[str, TestSession] = {}

    async with OpenMBTIClient() as mcp_client:
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Testing: {model}")

            output_path = get_model_output_path(output_dir, model)

            existing_session = None
            if continue_run:
                existing_session = load_existing_session(output_path)

            session = await run_model_tests(
                model=model,
                num_runs=num_runs,
                api_key=api_key,
                output_path=output_path,
                parallel=parallel,
                mcp_client=mcp_client,
                temperature=temperature,
                scale=scale,
                existing_session=existing_session,
            )

            results[model] = session

            # Print model summary
            if session.runs:
                summary = session.to_dict()["summary"]
                dist = summary["mbti_distribution"]
                dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
                print(f"  Summary: {dist_str}")

    return results


def print_final_summary(results: dict[str, TestSession], output_dir: Path):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    summary_data = []

    for model, session in results.items():
        if not session.runs:
            continue

        summary = session.to_dict()["summary"]
        dist = summary["mbti_distribution"]

        # Find most common type
        most_common = max(dist.items(), key=lambda x: x[1]) if dist else ("N/A", 0)

        summary_data.append({
            "model": model,
            "completed": summary["completed_runs"],
            "total": session.total_runs,
            "most_common": most_common[0],
            "distribution": dist,
        })

        short_model = model.split("/")[-1]
        print(f"\n{short_model}:")
        print(f"  Completed: {summary['completed_runs']}/{session.total_runs}")
        print(f"  Most common: {most_common[0]} ({most_common[1]}x)")
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  Distribution: {dist_str}")

    # Save summary JSON
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary saved to: {summary_path}")


def main():
    load_dotenv()

    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = get_output_dir(args.run_id)

    print("=" * 60)
    print("LLM Personality Test - Batch Run")
    print("=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Models: {len(args.models)}")
    print(f"Runs per model: {args.runs}")
    print(f"Parallel: {args.parallel}")
    print(f"Temperature: {args.temperature}")
    print(f"Scale: {args.scale}-point")
    print(f"Output: {output_dir}/")
    if args.continue_run:
        print("Mode: Continue from existing")

    results = asyncio.run(
        run_all_models(
            models=args.models,
            num_runs=args.runs,
            api_key=api_key,
            output_dir=output_dir,
            parallel=args.parallel,
            continue_run=args.continue_run,
            temperature=args.temperature,
            scale=args.scale,
        )
    )

    print_final_summary(results, output_dir)


if __name__ == "__main__":
    main()
