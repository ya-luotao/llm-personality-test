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
from src.models import TestSession, TestResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MBTI personality tests on LLMs via OpenMBTI MCP server"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of test runs (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: results/<model>_<timestamp>.json)",
    )
    parser.add_argument(
        "--continue-from",
        "-c",
        dest="continue_from",
        help="Continue from an existing results JSON file",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=3,
        help="Number of parallel runs (default: 3)",
    )
    return parser.parse_args()


def get_output_path(model: str, output: str | None) -> Path:
    if output:
        return Path(output)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{safe_model}_{timestamp}.json"


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
            from src.models import QuestionAnswer
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
        print(f"Warning: Could not load existing session: {e}", file=sys.stderr)
        return None


def save_session(session: TestSession, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)


async def run_single_test_with_output(
    runner: MBTITestRunner,
    run_id: int,
    total_runs: int,
    session: TestSession,
    output_path: Path,
    lock: asyncio.Lock,
) -> TestResult | None:
    try:
        result = await runner.run_single_test(run_id=run_id)

        async with lock:
            session.runs.append(result)
            save_session(session, output_path)

            completed = len(session.runs)
            print(f"\n[Run {run_id}/{total_runs}] Completed: {result.mbti_type}")
            print(f"  Progress: {completed}/{total_runs} runs saved")

            print(f"  Choices: ", end="")
            choices = [str(q.llm_choice) for q in result.questions[:10]]
            print(" ".join(choices) + ("..." if len(result.questions) > 10 else ""))

        return result
    except Exception as e:
        async with lock:
            print(f"\n[Run {run_id}/{total_runs}] Error: {e}", file=sys.stderr)
        return None


async def run_tests(
    model: str,
    num_runs: int,
    api_key: str,
    output_path: Path,
    parallel: int,
    existing_session: TestSession | None = None,
) -> TestSession:
    if existing_session:
        session = existing_session
        session.total_runs = num_runs
        completed_run_ids = {r.run_id for r in session.runs}
        remaining_run_ids = [i for i in range(1, num_runs + 1) if i not in completed_run_ids]
        print(f"Resuming: {len(session.runs)} completed, {len(remaining_run_ids)} remaining")
    else:
        session = TestSession(
            model=model,
            total_runs=num_runs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        remaining_run_ids = list(range(1, num_runs + 1))

    if not remaining_run_ids:
        print("All runs already completed!")
        return session

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(parallel)

    async def bounded_run(mcp_client: OpenMBTIClient, run_id: int):
        async with semaphore:
            llm_client = LLMClient(model=model, api_key=api_key)
            runner = MBTITestRunner(mcp_client, llm_client)
            return await run_single_test_with_output(
                runner, run_id, num_runs, session, output_path, lock
            )

    async with OpenMBTIClient() as mcp_client:
        tasks = [bounded_run(mcp_client, run_id) for run_id in remaining_run_ids]
        await asyncio.gather(*tasks)

    return session


def main():
    load_dotenv()

    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment", file=sys.stderr)
        print("Please set it in .env file or environment variable", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")
    print(f"Parallel: {args.parallel}")

    existing_session = None
    if args.continue_from:
        output_path = Path(args.continue_from)
        existing_session = load_existing_session(output_path)
        if existing_session:
            print(f"Continuing from: {output_path}")
        else:
            print(f"Starting fresh (could not load {output_path})")
    else:
        output_path = get_output_path(args.model, args.output)

    print(f"Output: {output_path}")

    session = asyncio.run(
        run_tests(
            args.model,
            args.runs,
            api_key,
            output_path,
            args.parallel,
            existing_session,
        )
    )

    print(f"\nResults saved to: {output_path}")

    if session.runs:
        print("\nSummary:")
        summary = session.to_dict()["summary"]
        print(f"  Completed: {summary['completed_runs']}/{args.runs}")
        print(f"  MBTI Distribution: {summary['mbti_distribution']}")


if __name__ == "__main__":
    main()
