"""Run a self-contained evaluation without overwriting the published Groq snapshot.

Default new-run outputs live under experiments/runs/<provider>__<model>/.
Use --fresh to restart a run in that directory. The original
experiments/results/*.csv files remain immutable historical evidence.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from llm_client import LLMClient          # noqa: E402
from prompting_strategies import STRATEGIES  # noqa: E402
from evaluator import evaluate_sample      # noqa: E402
from metrics import summarize_by_strategy  # noqa: E402

DATASET_PATH = BASE_DIR / "data" / "generated_dataset.json"
RESULTS_DIR = BASE_DIR / "experiments" / "results"
# Published results are intentionally not the default write destination.



def load_dataset(limit: int = None, dataset_path=None):
    path = Path(dataset_path) if dataset_path else DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def load_existing_results(fresh: bool, raw_path: Path, provider: str, model: str):
    """Returns (existing_df, completed_keys_set). Empty if --fresh or no prior file."""
    if fresh or not raw_path.exists():
        return pd.DataFrame(), set()

    existing_df = pd.read_csv(raw_path)
    required = {"parse_success", "provider", "model", "sample_id", "strategy"}
    if not required.issubset(existing_df.columns):
        raise ValueError("Existing results lack provider/model/run metadata; use --fresh")
    if not existing_df.empty and not (existing_df["provider"].eq(provider) & existing_df["model"].eq(model)).all():
        raise ValueError("Refusing to mix different providers or models in one results directory")

    # Only treat SUCCESSFULLY parsed rows as "done" -- anything that failed
    # (e.g. due to a quota error) should be retried by the next provider.
    completed = existing_df[existing_df["parse_success"] == True]
    completed_keys = set(zip(completed["sample_id"], completed["strategy"]))
    return existing_df, completed_keys


def run(provider: str = None, limit: int = None, sleep_between_calls: float = 0.0,
        fresh: bool = False, output_dir: str = None, dataset_path: str = None):
    dataset = load_dataset(limit, dataset_path)
    client = LLMClient(provider=provider)

    run_name = "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in f"{client.provider}__{client.model}")
    results_dir = Path(output_dir).resolve() if output_dir else BASE_DIR / "experiments" / "runs" / run_name
    if results_dir.resolve() == RESULTS_DIR.resolve():
        raise ValueError("Published experiments/results is read-only; choose a separate --output-dir")
    raw_path = results_dir / "raw_results.csv"
    summary_path = results_dir / "summary_metrics.csv"
    existing_df, completed_keys = load_existing_results(fresh, raw_path, client.provider, client.model)
    if len(completed_keys) > 0:
        print(f"Resuming: {len(completed_keys)} (sample, strategy) pairs already completed — skipping those.")

    print(f"Provider: {client.provider} | Model: {client.model}")
    n_clean = sum(1 for e in dataset if e.get("has_error", True) is False)
    print(f"Samples: {len(dataset)} ({len(dataset) - n_clean} buggy, {n_clean} clean) | Strategies: {list(STRATEGIES.keys())}")
    total_possible = len(dataset) * len(STRATEGIES)
    remaining = total_possible - len(completed_keys)
    print(f"Total (sample, strategy) pairs: {total_possible} | Remaining to run: {remaining}")
    print()

    new_records = []
    done = 0

    for entry in dataset:
        for strategy_name, prompt_fn in STRATEGIES.items():
            key = (entry["sample_id"], strategy_name)
            if key in completed_keys:
                continue  # already have a good result for this pair

            prompt = prompt_fn(entry["buggy_code"])
            try:
                raw_response = client.generate(prompt)
            except Exception as e:
                raw_response = json.dumps({"_error": str(e)})
                print(f"  ! Call failed for {key} ({client.provider}): {e}")

            record = evaluate_sample(entry, strategy_name, raw_response)
            record["provider"] = client.provider
            record["model"] = client.model
            new_records.append(record)

            done += 1
            if done % 10 == 0 or done == remaining:
                print(f"  Progress: {done}/{remaining}")

            if sleep_between_calls:
                time.sleep(sleep_between_calls)

    results_dir.mkdir(parents=True, exist_ok=True)

    if len(new_records) == 0:
        print("\nNothing new to run — all (sample, strategy) pairs already completed.")
        final_df = existing_df
    else:
        new_df = pd.DataFrame(new_records)
        if not existing_df.empty:
            # Drop old (failed) rows for keys we just retried, then append new results.
            new_keys = set(zip(new_df["sample_id"], new_df["strategy"]))
            existing_df = existing_df[
                ~existing_df.apply(lambda r: (r["sample_id"], r["strategy"]) in new_keys, axis=1)
            ]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

    final_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved -> {raw_path} ({len(final_df)} total rows)")

    if "provider" in final_df.columns:
        print("Providers used so far:", final_df["provider"].value_counts().to_dict())

    summary_df = summarize_by_strategy(final_df)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved -> {summary_path}\n")
    print(summary_df.to_string(index=False))

    ok_df = final_df[final_df["parse_success"] == True]
    still_missing = total_possible - len(set(zip(ok_df["sample_id"], ok_df["strategy"])))
    if still_missing > 0:
        print(f"\n{still_missing} (sample, strategy) pairs still not successfully completed.")
        print("Re-run with the same provider and model to retry unfinished evaluations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default=None,
                         help="Override PYSYNGUARD_PROVIDER for this run (gemini, openai, anthropic, groq, mock)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of dataset samples (useful for quick tests)")
    parser.add_argument("--sleep", type=float, default=0.0,
                         help="Seconds to sleep between API calls (rate limiting)")
    parser.add_argument("--output-dir", default=None,
                         help="Directory for this run (default: experiments/runs/<provider>__<model>)")
    parser.add_argument("--dataset", default=None,
                         help="Dataset JSON to evaluate (default: data/generated_dataset.json; "
                              "use data/dataset_with_clean.json to include the 10 clean samples)")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing results and start completely from scratch")
    args = parser.parse_args()

    run(provider=args.provider, limit=args.limit, sleep_between_calls=args.sleep, fresh=args.fresh, output_dir=args.output_dir,
        dataset_path=args.dataset)
