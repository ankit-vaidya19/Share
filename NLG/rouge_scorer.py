"""
Script for computing ROUGE-L scores on evaluation results.
Processes CSV files containing predictions and ground truth labels.
"""

import argparse
import os
import string
from typing import List, Callable

import pandas as pd
from rouge_score import rouge_scorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-L scores for evaluation results"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing result CSV files (e.g., ./results/train or ./results/eval)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save scores summary. If not provided, prints to stdout.",
    )

    return parser.parse_args()


def normalize_answer(s: str) -> str:
    """
    Normalize text by lowercasing and removing punctuation/extra whitespace.

    Args:
        s: Input string.

    Returns:
        Normalized string.
    """

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def metric_max_over_ground_truths(
    metric_fn: Callable,
    prediction: str,
    ground_truths: List[str],
) -> float:
    """
    Compute the maximum metric score over multiple ground truths.

    Args:
        metric_fn: Scoring function.
        prediction: Model prediction.
        ground_truths: List of acceptable ground truth answers.

    Returns:
        Maximum score across all ground truths.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def rougeL_score(prediction: str, ground_truth: str) -> float:
    """
    Compute ROUGE-L F1 score between prediction and ground truth.

    Args:
        prediction: Model prediction string.
        ground_truth: Ground truth string.

    Returns:
        ROUGE-L F1 score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def compute_scores_for_file(file_path: str) -> float:
    """
    Compute average ROUGE-L score for a single result file.

    Args:
        file_path: Path to CSV file with 'preds' and 'gts' columns.

    Returns:
        Average ROUGE-L score as percentage.
    """
    df = pd.read_csv(file_path)
    predictions = df["preds"].tolist()
    references = df["gts"].tolist()

    rougeL = 0
    for pred, gold in zip(predictions, references):
        # Handle None/NaN values
        pred_str = str(pred) if pd.notna(pred) else ""
        gold_str = str(gold) if pd.notna(gold) else ""
        rougeL += rougeL_score(pred_str, gold_str)

    rougeL = 100.0 * rougeL / len(references)
    return rougeL


def main():
    args = parse_args()

    print("=" * 60)
    print("ROUGE-L Score Computation")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print("=" * 60)

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory '{args.results_dir}' not found")
        return

    # Get all CSV files
    files = [f for f in os.listdir(args.results_dir) if f.endswith(".csv")]

    if not files:
        print(f"No CSV files found in {args.results_dir}")
        return

    results = []
    for file in sorted(files):
        file_path = os.path.join(args.results_dir, file)
        score = compute_scores_for_file(file_path)
        task_name = file.replace(".csv", "")
        results.append({"task": task_name, "rouge_l": score})
        print(f"{task_name}: {score:.2f}")

    # Compute average
    avg_score = sum(r["rouge_l"] for r in results) / len(results)
    print("-" * 40)
    print(f"Average ROUGE-L: {avg_score:.2f}")

    # Save results if requested
    if args.output_file:
        results_df = pd.DataFrame(results)
        results_df.loc[len(results_df)] = {"task": "AVERAGE", "rouge_l": avg_score}
        results_df.to_csv(args.output_file, index=False)
        print(f"Saved results to {args.output_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
