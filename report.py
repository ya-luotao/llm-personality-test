#!/usr/bin/env python3
"""Analyze MBTI test results by run ID."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MBTI test results")
    parser.add_argument(
        "run_id",
        help="Run ID (folder name in results/)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Specific model to analyze (default: all models)",
    )
    return parser.parse_args()


def load_results(run_id: str, model: str | None = None) -> dict[str, dict]:
    """Load all result files for a run ID."""
    results_dir = Path("results") / run_id

    if not results_dir.exists():
        # Try as direct file path
        if Path(run_id).exists():
            with open(run_id, "r") as f:
                data = json.load(f)
            return {data["model"]: data}
        print(f"Error: Run ID '{run_id}' not found", file=sys.stderr)
        sys.exit(1)

    results = {}
    for file in results_dir.glob("*.json"):
        if file.name.startswith("_"):
            continue
        with open(file, "r") as f:
            data = json.load(f)
        model_name = data.get("model", file.stem)
        if model and model not in model_name:
            continue
        results[model_name] = data

    if not results:
        print(f"Error: No results found for run ID '{run_id}'", file=sys.stderr)
        sys.exit(1)

    return results


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


def analyze_mbti_distribution(data: dict) -> dict[str, int]:
    """Get MBTI type distribution from runs."""
    types = [run["mbti_type"] for run in data.get("runs", [])]
    return dict(Counter(types).most_common())


def analyze_consistency(data: dict) -> dict:
    """Analyze choice consistency across runs."""
    runs = data.get("runs", [])
    if not runs:
        return {}

    # Get all questions from first run
    questions = runs[0].get("questions", [])
    num_questions = len(questions)

    consistency = []
    for q_idx in range(num_questions):
        choices = []
        for run in runs:
            if q_idx < len(run.get("questions", [])):
                choices.append(run["questions"][q_idx]["llm_choice"])

        if choices:
            most_common = Counter(choices).most_common(1)[0]
            consistency.append({
                "question_idx": q_idx,
                "text": questions[q_idx]["text"] if q_idx < len(questions) else f"Q{q_idx+1}",
                "choices": choices,
                "most_common": most_common[0],
                "agreement": most_common[1] / len(choices) * 100,
                "unique_choices": len(set(choices)),
            })

    return {
        "by_question": consistency,
        "avg_agreement": sum(c["agreement"] for c in consistency) / len(consistency) if consistency else 0,
    }


def analyze_dimensions(data: dict) -> dict:
    """Analyze dimension scores across runs."""
    runs = data.get("runs", [])
    if not runs:
        return {}

    dimensions = {"E_I": [], "S_N": [], "T_F": [], "J_P": []}
    dim_labels = {
        "E_I": ("E", "I"),
        "S_N": ("S", "N"),
        "T_F": ("T", "F"),
        "J_P": ("J", "P"),
    }

    for run in runs:
        scores = run.get("dimension_scores", {})
        for dim in dimensions:
            if dim in scores:
                # Get the dominant trait percentage
                labels = dim_labels[dim]
                first_score = scores[dim].get(labels[0], 50)
                dimensions[dim].append(first_score)

    result = {}
    for dim, scores in dimensions.items():
        if scores:
            labels = dim_labels[dim]
            avg = sum(scores) / len(scores)
            dominant = labels[0] if avg >= 50 else labels[1]
            result[dim] = {
                "avg_first": avg,
                "avg_second": 100 - avg,
                "dominant": dominant,
                "scores": scores,
                "std_dev": (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5,
            }

    return result


def print_choice_table(data: dict, max_runs: int = 10):
    """Print a table of choices per question across runs."""
    runs = data.get("runs", [])
    if not runs:
        return

    questions = runs[0].get("questions", [])
    num_runs = min(len(runs), max_runs)

    # Header
    header = f"{'Question':<30} |"
    for i in range(num_runs):
        header += f" R{i+1}"
    header += " | Mode | Agr%"
    print(header)
    print("-" * len(header))

    # Rows
    for q_idx, q in enumerate(questions):
        text = q["text"][:28] + ".." if len(q["text"]) > 30 else q["text"]
        row = f"{text:<30} |"

        choices = []
        for run_idx in range(num_runs):
            if run_idx < len(runs) and q_idx < len(runs[run_idx].get("questions", [])):
                choice = runs[run_idx]["questions"][q_idx]["llm_choice"]
                choices.append(choice)
                row += f"  {choice}"
            else:
                row += "  -"

        if choices:
            most_common = Counter(choices).most_common(1)[0]
            agreement = most_common[1] / len(choices) * 100
            row += f" |  {most_common[0]}   | {agreement:4.0f}%"

        print(row)


def print_consistency_summary(consistency: dict):
    """Print consistency analysis summary."""
    by_question = consistency.get("by_question", [])
    if not by_question:
        return

    # Sort by agreement (lowest first = most inconsistent)
    sorted_questions = sorted(by_question, key=lambda x: x["agreement"])

    print(f"\nOverall consistency: {consistency['avg_agreement']:.1f}%")

    # Most inconsistent questions
    print("\nMost inconsistent questions:")
    for q in sorted_questions[:5]:
        choices_str = " ".join(str(c) for c in q["choices"])
        print(f"  {q['agreement']:5.1f}% | {q['text'][:40]:<40} | {choices_str}")

    # Most consistent questions
    print("\nMost consistent questions:")
    for q in sorted_questions[-5:]:
        choices_str = " ".join(str(c) for c in q["choices"])
        print(f"  {q['agreement']:5.1f}% | {q['text'][:40]:<40} | {choices_str}")


def print_dimension_analysis(dimensions: dict):
    """Print dimension score analysis."""
    dim_names = {
        "E_I": "Extraversion / Introversion",
        "S_N": "Sensing / Intuition",
        "T_F": "Thinking / Feeling",
        "J_P": "Judging / Perceiving",
    }

    print(f"\n{'Dimension':<30} | {'Dominant':<8} | {'Avg %':<6} | {'Std Dev':<7} | Scores")
    print("-" * 80)

    for dim, analysis in dimensions.items():
        name = dim_names.get(dim, dim)
        scores_str = " ".join(f"{s:.0f}" for s in analysis["scores"][:10])
        print(
            f"{name:<30} | {analysis['dominant']:<8} | "
            f"{analysis['avg_first']:5.1f}% | {analysis['std_dev']:6.2f}  | {scores_str}"
        )


def print_model_comparison(results: dict[str, dict]):
    """Print comparison across multiple models."""
    if len(results) < 2:
        return

    print_header("MODEL COMPARISON")

    # MBTI distribution comparison
    print(f"\n{'Model':<35} | {'Runs':<5} | {'Types':<30} | Consistency")
    print("-" * 90)

    for model, data in sorted(results.items()):
        short_model = model.split("/")[-1][:33]
        runs = len(data.get("runs", []))
        dist = analyze_mbti_distribution(data)
        consistency = analyze_consistency(data)

        types_str = ", ".join(f"{t}:{c}" for t, c in list(dist.items())[:4])
        avg_consistency = consistency.get("avg_agreement", 0)

        print(f"{short_model:<35} | {runs:<5} | {types_str:<30} | {avg_consistency:.1f}%")

    # Dimension comparison
    print_subheader("Dimension Tendencies")
    print(f"\n{'Model':<25} | {'E/I':<12} | {'S/N':<12} | {'T/F':<12} | {'J/P':<12}")
    print("-" * 80)

    for model, data in sorted(results.items()):
        short_model = model.split("/")[-1][:23]
        dims = analyze_dimensions(data)

        def fmt_dim(d):
            if d in dims:
                return f"{dims[d]['dominant']} ({dims[d]['avg_first']:.0f}%)"
            return "-"

        print(
            f"{short_model:<25} | {fmt_dim('E_I'):<12} | {fmt_dim('S_N'):<12} | "
            f"{fmt_dim('T_F'):<12} | {fmt_dim('J_P'):<12}"
        )


def analyze_single_model(model: str, data: dict):
    """Analyze a single model's results."""
    short_model = model.split("/")[-1]
    print_header(f"ANALYSIS: {short_model}")

    runs = data.get("runs", [])
    print(f"\nModel: {model}")
    print(f"Total runs: {len(runs)}")

    # MBTI distribution
    print_subheader("MBTI Type Distribution")
    dist = analyze_mbti_distribution(data)
    for mbti_type, count in dist.items():
        pct = count / len(runs) * 100 if runs else 0
        bar = "#" * int(pct / 5)
        print(f"  {mbti_type}: {count:3d} ({pct:5.1f}%) {bar}")

    # Choice table
    print_subheader("Choices by Question")
    print_choice_table(data)

    # Consistency analysis
    print_subheader("Consistency Analysis")
    consistency = analyze_consistency(data)
    print_consistency_summary(consistency)

    # Dimension analysis
    print_subheader("Dimension Scores")
    dimensions = analyze_dimensions(data)
    print_dimension_analysis(dimensions)


def main():
    args = parse_args()

    results = load_results(args.run_id, args.model)

    print_header(f"LLM Personality Test Report: {args.run_id}")
    print(f"Models analyzed: {len(results)}")

    # Analyze each model
    for model, data in sorted(results.items()):
        analyze_single_model(model, data)

    # Compare models if multiple
    if len(results) > 1:
        print_model_comparison(results)

    print()


if __name__ == "__main__":
    main()
