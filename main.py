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
from src.models import TestSession


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
    return parser.parse_args()


def get_output_path(model: str, output: str | None) -> Path:
    if output:
        return Path(output)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{safe_model}_{timestamp}.json"


async def run_tests(model: str, num_runs: int, api_key: str) -> TestSession:
    session = TestSession(
        model=model,
        total_runs=num_runs,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    async with OpenMBTIClient() as mcp_client:
        llm_client = LLMClient(model=model, api_key=api_key)
        runner = MBTITestRunner(mcp_client, llm_client)

        for i in range(num_runs):
            print(f"\nRun {i + 1}/{num_runs}...")
            try:
                result = await runner.run_single_test(run_id=i + 1)
                session.runs.append(result)
                print(f"  Result: {result.mbti_type}")
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

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

    session = asyncio.run(run_tests(args.model, args.runs, api_key))

    output_path = get_output_path(args.model, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    if session.runs:
        print("\nSummary:")
        summary = session.to_dict()["summary"]
        print(f"  Completed: {summary['completed_runs']}/{args.runs}")
        print(f"  MBTI Distribution: {summary['mbti_distribution']}")


if __name__ == "__main__":
    main()
