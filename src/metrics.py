"""
metrics.py
----------
Computes aggregate metrics from a list of per-sample evaluation records
(as produced by evaluator.py) so strategies can be compared.
"""

import pandas as pd


def compute_metrics(records: list) -> dict:
    """
    records: list of dicts, each with at least:
        - ground_truth_has_error (bool)
        - predicted_has_error (bool)
        - ground_truth_line (int or None)
        - predicted_line (int or None)
        - parse_success (bool)  # whether the LLM's JSON response was parseable

    Returns a dict of aggregate metrics.
    """
    n = len(records)
    if n == 0:
        return {}

    tp = fp = tn = fn = 0
    correct_localization = 0
    localizable_cases = 0  # cases where both ground truth AND prediction say "has error"
    parse_failures = 0

    for r in records:
        if not r.get("parse_success", True):
            parse_failures += 1
            # Count an unparseable response as a failed verdict, not as a
            # silently excluded sample (a miss for buggy code, a false alarm for clean code).
            if r["ground_truth_has_error"]:
                fn += 1
            else:
                fp += 1
            continue

        gt = r["ground_truth_has_error"]
        pred = r["predicted_has_error"]

        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif not gt and not pred:
            tn += 1
        elif gt and not pred:
            fn += 1

        if gt and pred:
            localizable_cases += 1
            if r.get("ground_truth_line") == r.get("predicted_line"):
                correct_localization += 1

    total_scored = tp + fp + tn + fn
    accuracy = (tp + tn) / total_scored if total_scored else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    localization_accuracy = (correct_localization / localizable_cases) if localizable_cases else 0.0
    # Specificity (true-negative rate) needs clean samples; undefined otherwise.
    specificity = tn / (tn + fp) if (tn + fp) else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None

    return {
        "n_samples": n,
        "parse_failures": parse_failures,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "localization_accuracy": round(localization_accuracy, 4),
        "false_positive_rate": round(false_positive_rate, 4) if false_positive_rate is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
    }


def summarize_by_strategy(raw_results_df: pd.DataFrame) -> pd.DataFrame:
    """Groups raw per-sample results by strategy and computes metrics for each."""
    summaries = []
    for strategy, group in raw_results_df.groupby("strategy"):
        records = group.to_dict("records")
        # Adapt column names to what compute_metrics expects
        adapted = [{
            "ground_truth_has_error": r["ground_truth_has_error"],
            "predicted_has_error": r["predicted_has_error"],
            "ground_truth_line": r["ground_truth_line"],
            "predicted_line": r["predicted_line"],
            "parse_success": r["parse_success"],
        } for r in records]
        metrics = compute_metrics(adapted)
        metrics["strategy"] = strategy
        summaries.append(metrics)

    cols = ["strategy", "n_samples", "true_positives", "false_negatives",
            "true_negatives", "false_positives", "accuracy", "precision", "recall",
            "f1_score", "localization_accuracy", "false_positive_rate", "specificity",
            "parse_failures"]
    df = pd.DataFrame(summaries)
    return df[cols].sort_values("f1_score", ascending=False).reset_index(drop=True)
