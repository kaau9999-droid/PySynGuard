# PySynGuard — LLM-Based Python Syntax Error Detection

A course-scale experiment comparing **zero-shot, few-shot, chain-of-thought (CoT), and role-based prompting** for Python syntax-error detection and injected-line localization. The proposed Agile/Extreme Programming (XP) CI integration is a *design scenario*, **not a deployed or benchmarked CI integration**.

**Course:** MSCS811 — Advanced Software Design and Development, Taibah University.

## Published experiment snapshot

The included `experiments/results/` directory contains the original, **saved** Groq experiment (not a new live run): **66 buggy snippets × 4 prompt strategies = 264 records**, model `openai/gpt-oss-120b`, served by Groq. The originals are retained for reproducibility and matching the course report.

| Prompt strategy | Detected / 66 | Detection recall | Exact injected-line localization among detections |
|---|---:|---:|---:|
| Zero-shot | 65 | 98.48% | 64/65 (98.46%) |
| Few-shot | 66 | 100% | 65/66 (98.48%) |
| CoT | 66 | 100% | 63/66 (95.45%) |
| Role-based | 65 | 98.48% | 63/65 (96.92%) |

**Interpretation:** Every snippet in this original run is buggy, so there are no clean negative cases. Consequently **false-positive rate and specificity are undefined (N/A)** and the precision/F1 values are limited in interpretation. The original `summary_metrics.csv` historically encoded FPR as `0`, which is an undefined-denominator placeholder, **not a measured zero false-positive rate**. The dashboard presents it as N/A without changing the historical CSV. Python `ast.parse()` confirms that snippets do not parse; it does not independently validate the injected category/line. The single-snippet UI compares against the parser-reported line, while batch results compare against the injector-recorded line; these need not agree. Recorded responses do not prove that newly repeated runs will return the same predictions.

## Repository contents

```
 data/clean_samples/         10 small, syntactically valid Python source files
 data/error_types.json       Eight synthetic mutation types
 data/generated_dataset.json 66 saved corrupted snippets
 data/dataset_with_clean.json 76 snippets: the 66 above + 10 clean files (has_error=false)
 src/error_injector.py       Recreate the generated dataset (seed 42)
 src/ground_truth_checker.py Python AST parser check
 src/prompting_strategies.py Four prompting strategies
 src/llm_client.py          Providers: Groq, Gemini, OpenAI, Anthropic, mock
 src/evaluator.py           JSON response validation and scoring
 src/metrics.py             Aggregate evaluation metrics
 experiments/run_experiment.py
 experiments/results/       ORIGINAL SAVED CSVs (do not overwrite)
 notebooks/                 Analysis notebook and plots
 ui/app.py                  Streamlit UI and historical experiment dashboard
```

## Quick start (Windows / VS Code PowerShell)

From the project root (where `requirements.txt` is located):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Open `http://localhost:8501`. In **Try It Yourself**, choose `groq` to use a real Groq model or `mock` for offline interface testing. The latter is simulated and **must never be represented as real LLM evaluation evidence**. The **Experiment Dashboard** displays the supplied historical CSV results, and does not initiate any API calls.

### Groq key (never commit it)

Create your own key at Groq Console. Either set it in the **same PowerShell Terminal session** before launching the UI:

```powershell
$env:GROQ_API_KEY = (Read-Host "Enter your Groq API key")
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Or copy `.env.example` to `.env`, set `PYSYNGUARD_PROVIDER=groq` and `GROQ_API_KEY=...` *locally*. **Never commit `.env`, paste the key into source files, or share terminal screenshots exposing secrets.** The key is not bundled here.

## Reproducing the dataset

```powershell
.\.venv\Scripts\python.exe src/error_injector.py
```

The published dataset is already bundled; regeneration uses the fixed seed. Back up the original if you want to compare regenerated data.

## Running a NEW experiment (optional; may incur API cost)

```powershell
# Offline pipeline smoke test — 2 samples, 4 prompts; NOT report results
.\.venv\Scripts\python.exe experiments/run_experiment.py --provider mock --limit 2 --fresh

# Live Groq run — 66 samples, 4 strategies (264 evaluations)
.\.venv\Scripts\python.exe experiments/run_experiment.py --provider groq --fresh
```

New results are written by default under `experiments/runs/<provider>__<model>/`, **not** to the original `experiments/results/` snapshot. Re-running without `--fresh` resumes the current provider/model run. You can set `--output-dir` to another non-published directory. The runner refuses to mix providers/models in the same results file; compare providers using separately collected runs. `experiments/runs/` is ignored by Git, so explicitly export any results you decide to publish. Individual live calls require your own provider account and may be rate-limited or billable.

For a real evaluation, do **not** claim FPR or specificity without clean snippets. Failed/unparseable model responses are treated as unsuccessful predictions when computing metrics; they are recorded separately as parse failures. The historical results are not retroactively changed by these fixes.

## Extended run with clean samples (false-positive rate)

The original run cannot measure false alarms. `data/dataset_with_clean.json` adds the 10 clean source files as negative samples (`C001`–`C010`, `has_error: false`, `error_line: null`). The evaluator now reads this label, so a "no error" answer on a clean file counts as a true negative, and the summary reports specificity and FPR.

To avoid repeating the 264 saved calls, copy the original results into the run folder; the runner then skips them and makes only the 40 new calls (10 clean samples × 4 strategies), using the same Groq model:

```powershell
mkdir experiments\runs\groq__openai_gpt-oss-120b
copy experiments\results\raw_results.csv experiments\runs\groq__openai_gpt-oss-120b\
.\.venv\Scripts\python.exe experiments/run_experiment.py --provider groq --dataset data/dataset_with_clean.json
```

To publish the extended run, copy the two CSV files from `experiments/runs/groq__openai_gpt-oss-120b/` to `experiments/results_with_clean/`. The dashboard then offers both saved runs. The original `experiments/results/` is left unchanged.

## Limitations and future work

The dataset contains only synthetic errors from 10 handwritten source files; categories have unequal sample sizes and samples are related. The four prompts were compared using a single model with no repeated-run variability analysis, and the extended run has only 10 clean samples, so its false-positive estimate is coarse. No developer study, latency/cost benchmark, nor actual CI integration was evaluated. Useful extensions include more (and more varied) clean-code cases, parser-vs-injection localization reporting, larger authentic projects, repeated evaluations, and an implemented pre-commit/CI benchmark.
