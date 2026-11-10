"""
error_injector.py
------------------
Takes clean Python source files and injects a single, known syntax error
into each one according to the taxonomy in data/error_types.json.

Produces data/generated_dataset.json where each entry contains:
    - sample_id
    - source_file
    - clean_code
    - buggy_code
    - error_type_id
    - error_line (1-indexed line number where the error was injected)
    - is_valid_check (sanity check that buggy_code actually fails ast.parse)
"""

import ast
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean_samples"
ERROR_TYPES_PATH = BASE_DIR / "data" / "error_types.json"
OUTPUT_PATH = BASE_DIR / "data" / "generated_dataset.json"

random.seed(42)  # reproducibility


def load_error_types():
    with open(ERROR_TYPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["error_types"]


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---- Individual injector functions ----
# Each takes the list of source lines and returns (new_lines, line_index) or None if not applicable.

def inject_missing_colon(lines):
    candidates = [i for i, l in enumerate(lines) if re.search(r":\s*$", l.rstrip())
                  and re.match(r"^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\b", l)]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    new_lines[idx] = re.sub(r":\s*$", "", new_lines[idx].rstrip("\n")) + "\n"
    return new_lines, idx


def inject_unmatched_bracket(lines):
    candidates = [i for i, l in enumerate(lines) if ")" in l or "]" in l or "}" in l]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    line = new_lines[idx]
    for ch in [")", "]", "}"]:
        if ch in line:
            pos = line.rfind(ch)
            new_lines[idx] = line[:pos] + line[pos + 1:]
            break
    return new_lines, idx


def inject_bad_indentation(lines):
    candidates = [i for i, l in enumerate(lines) if l.startswith("    ") and l.strip() != ""]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    # Strip indentation entirely on a line that is supposed to be nested,
    # which breaks the block structure relative to surrounding lines.
    new_lines[idx] = new_lines[idx].lstrip()
    return new_lines, idx


def inject_missing_parenthesis_call(lines):
    candidates = [i for i, l in enumerate(lines) if re.search(r"print\((.*)\)", l)]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    new_lines[idx] = re.sub(r"print\((.*)\)", r"print \1", new_lines[idx])
    return new_lines, idx


def inject_unterminated_string(lines):
    candidates = [i for i, l in enumerate(lines) if re.search(r'"[^"]*"', l)]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    line = new_lines[idx]
    last_quote = line.rfind('"')
    new_lines[idx] = line[:last_quote] + line[last_quote + 1:]
    return new_lines, idx


def inject_invalid_operator(lines):
    candidates = [i for i, l in enumerate(lines) if re.search(r"\+|\-", l) and "=" in l]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    new_lines[idx] = re.sub(r"\+", "+*", new_lines[idx], count=1)
    if new_lines[idx] == lines[idx]:
        return None
    return new_lines, idx


def inject_missing_equals(lines):
    candidates = [i for i, l in enumerate(lines) if re.match(r"^\s*\w+\s=\s[^=]", l) and "==" not in l]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    new_lines[idx] = re.sub(r"(\w+)\s=\s", r"\1  ", new_lines[idx], count=1)
    return new_lines, idx


def inject_dangling_comma_paren(lines):
    candidates = [i for i, l in enumerate(lines) if l.strip() != "" and not l.strip().endswith((":", "\\"))]
    if not candidates:
        return None
    idx = random.choice(candidates)
    new_lines = lines.copy()
    new_lines[idx] = new_lines[idx].rstrip("\n") + ")\n"
    return new_lines, idx


INJECTORS = {
    "missing_colon": inject_missing_colon,
    "unmatched_bracket": inject_unmatched_bracket,
    "bad_indentation": inject_bad_indentation,
    "missing_parenthesis_call": inject_missing_parenthesis_call,
    "unterminated_string": inject_unterminated_string,
    "invalid_operator": inject_invalid_operator,
    "missing_equals": inject_missing_equals,
    "dangling_comma_paren": inject_dangling_comma_paren,
}


def build_dataset(repeats_per_error_type=1):
    """
    For every clean sample x every error type, attempt an injection.
    repeats_per_error_type lets you generate more variety by re-running
    with different random choices (kept low to control API cost later).
    """
    error_types = load_error_types()
    clean_files = sorted(CLEAN_DIR.glob("*.py"))
    dataset = []
    sample_counter = 1

    for clean_file in clean_files:
        clean_code = clean_file.read_text(encoding="utf-8")
        lines = clean_code.splitlines(keepends=True)

        for error_type in error_types:
            injector_fn = INJECTORS.get(error_type["id"])
            if injector_fn is None:
                continue

            for _ in range(repeats_per_error_type):
                result = injector_fn(lines)
                if result is None:
                    continue  # this error type doesn't apply to this sample
                new_lines, error_line_idx = result
                buggy_code = "".join(new_lines)

                # Sanity check: injected code must actually be invalid,
                # and must differ from the original.
                if buggy_code == clean_code:
                    continue
                if is_valid_python(buggy_code):
                    continue  # injection didn't actually break syntax; skip

                dataset.append({
                    "sample_id": f"S{sample_counter:03d}",
                    "source_file": clean_file.name,
                    "clean_code": clean_code,
                    "buggy_code": buggy_code,
                    "error_type_id": error_type["id"],
                    "error_line": error_line_idx + 1,  # 1-indexed
                    "is_valid_check": is_valid_python(buggy_code),  # should always be False
                })
                sample_counter += 1

    return dataset


def main():
    dataset = build_dataset(repeats_per_error_type=1)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(dataset)} buggy samples -> {OUTPUT_PATH}")
    # Print a small distribution summary
    counts = {}
    for item in dataset:
        counts[item["error_type_id"]] = counts.get(item["error_type_id"], 0) + 1
    print("Distribution by error type:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
