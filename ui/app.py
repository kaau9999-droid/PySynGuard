"""
ui/app.py
---------
Streamlit interface for PySynGuard.

Run with:
    cd pysynguard
    streamlit run ui/app.py

Two tabs:
    1. Try It Yourself  - paste/generate code, run one strategy, see the result
    2. Experiment Dashboard - view the saved runs (66 buggy samples, and the
       optional 76-sample run that adds 10 clean samples)
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from llm_client import LLMClient                      # noqa: E402
from prompting_strategies import STRATEGIES            # noqa: E402
from ground_truth_checker import check_syntax          # noqa: E402
from evaluator import parse_llm_response               # noqa: E402

RESULTS_DIR = BASE_DIR / "experiments" / "results"
CLEAN_RESULTS_DIR = BASE_DIR / "experiments" / "results_with_clean"
DATASET_PATH = BASE_DIR / "data" / "generated_dataset.json"

st.set_page_config(page_title="PySynGuard", layout="wide")
st.title("🐍 PySynGuard — LLM-Based Python Syntax Error Detection")
st.caption(
    "Comparing prompting strategies (Zero-shot, Few-shot, Chain-of-Thought, Role-based) "
    "for detecting syntax errors — framed as a CI/pre-commit assistant for Agile/XP teams."
)

tab1, tab2 = st.tabs(["🔍 Try It Yourself", "📊 Experiment Dashboard"])

# ---------------------------------------------------------------------------
# TAB 1: Try It Yourself
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Run a single detection")

    default_code = 'def add(a, b)\n    return a + b\n\nprint(add(2, 3))'
    code_input = st.text_area("Paste Python code here:", value=default_code, height=200)

    col1, col2 = st.columns(2)
    with col1:
        strategy_choice = st.selectbox("Prompting strategy", list(STRATEGIES.keys()))
    with col2:
        provider_choice = st.selectbox(
            "LLM provider",
            ["mock", "gemini", "openai", "anthropic", "groq"],
            help="'mock' works with no API key, for demoing the UI. "
                 "Use a real provider (with its API key set as an env var) for real results.",
        )

    if st.button("Run Detection", type="primary"):
        gt = check_syntax(code_input)
        st.write("**Ground truth (via Python's `ast` module):**")
        st.json(gt)

        with st.spinner("Calling LLM..."):
            try:
                client = LLMClient(provider=provider_choice)
                prompt = STRATEGIES[strategy_choice](code_input)
                raw_response = client.generate(prompt)
                parsed = parse_llm_response(raw_response)

                st.write("**Raw LLM response:**")
                st.code(raw_response, language="json")

                if "_parse_error" not in parsed:
                    st.write("**Parsed result:**")
                    st.json(parsed)

                    if parsed["has_error"] == gt["has_error"]:
                        st.success("✅ Detection matches ground truth.")
                    else:
                        st.error("❌ Detection does NOT match ground truth.")

                    if gt["has_error"] and parsed["has_error"]:
                        if parsed["error_line"] == gt["error_line"]:
                            st.success(f"✅ Correct line ({gt['error_line']}).")
                        else:
                            st.warning(
                                f"⚠️ Line mismatch: LLM said {parsed['error_line']}, "
                                f"actual is {gt['error_line']}."
                            )
                else:
                    st.error("Could not parse the LLM's response as JSON.")

            except EnvironmentError as e:
                st.error(f"Missing API key: {e}. Set it as an environment variable, "
                          f"or choose 'mock' provider to test the UI without one.")
            except Exception as e:
                st.error(f"Error calling LLM: {e}")

# ---------------------------------------------------------------------------
# TAB 2: Experiment Dashboard
# ---------------------------------------------------------------------------
with tab2:
    run_options = {"Original run: 66 buggy samples": RESULTS_DIR}
    if (CLEAN_RESULTS_DIR / "summary_metrics.csv").exists():
        run_options["Extended run: 66 buggy + 10 clean samples"] = CLEAN_RESULTS_DIR
    run_label = st.radio("Saved run", list(run_options.keys()), horizontal=True)
    SUMMARY_PATH = run_options[run_label] / "summary_metrics.csv"
    RAW_RESULTS_PATH = run_options[run_label] / "raw_results.csv"
    st.subheader(f"Experiment results ({run_label})")

    if not SUMMARY_PATH.exists():
        st.warning(
            "No results found yet. Run the experiment first:\n\n"
            "```\ncd pysynguard\nPYSYNGUARD_PROVIDER=mock python3 experiments/run_experiment.py\n```"
        )
    else:
        summary_df = pd.read_csv(SUMMARY_PATH)
        raw_df = pd.read_csv(RAW_RESULTS_PATH)

        st.caption("Saved Groq run — these figures are loaded from CSV; viewing does not rerun the LLM.")
        # No clean samples exist in this saved run: FPR is undefined,
        # although the historical CSV stores 0 as a placeholder.
        if not (~raw_df["ground_truth_has_error"].astype(bool)).any():
            for col in ("false_positive_rate", "specificity"):
                if col in summary_df.columns:
                    summary_df[col] = float("nan")
        st.write("**Summary metrics by strategy:**")
        st.dataframe(summary_df.fillna("N/A"), width="stretch")

        st.write("**Accuracy comparison:**")
        st.bar_chart(summary_df.set_index("strategy")["accuracy"])

        st.write("**F1 Score comparison:**")
        st.bar_chart(summary_df.set_index("strategy")["f1_score"])

        st.write("**Localization accuracy comparison:**")
        st.bar_chart(summary_df.set_index("strategy")["localization_accuracy"])

        st.write("**Breakdown by error type:**")
        error_type_breakdown = (
            raw_df[raw_df["parse_success"]]
            .groupby(["strategy", "error_type_id"])["correct_detection"]
            .mean()
            .unstack()
        )
        st.dataframe(error_type_breakdown.style.format("{:.2f}"), width="stretch")

        with st.expander("🔎 View raw per-sample results"):
            st.dataframe(raw_df, width="stretch")

        st.download_button(
            "Download summary_metrics.csv",
            data=summary_df.to_csv(index=False),
            file_name="summary_metrics.csv",
        )
        st.download_button(
            "Download raw_results.csv",
            data=raw_df.to_csv(index=False),
            file_name="raw_results.csv",
        )
