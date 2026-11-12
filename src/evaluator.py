"""
evaluator.py
------------
Parses a raw LLM text response (expected to be JSON per prompting_strategies.py's
RESPONSE_FORMAT_INSTRUCTIONS) and compares it against the ground truth for a
given dataset sample, producing one evaluation record.
"""

import json
import re


def parse_llm_response(raw_text: str) -> dict:
    """
    Robustly extract a JSON object from the LLM's raw text response,
    tolerating markdown code fences or minor extra text around it.

    Returns {"has_error": bool, "error_line": int or None, "error_description": str}
    or {"_parse_error": True} if parsing fails entirely.
    """
    text = raw_text.strip()

    # Strip markdown fences if present
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    # If there's extra text, try to find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return {"_parse_error": True}
        flag = parsed.get("has_error")
        line = parsed.get("error_line")
        description = parsed.get("error_description")
        if type(flag) is not bool or not isinstance(description, str):
            return {"_parse_error": True}
        if flag:
            if type(line) is not int or line < 1:
                return {"_parse_error": True}
        elif line is not None:
            return {"_parse_error": True}
        return {"has_error": flag, "error_line": line, "error_description": description}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"_parse_error": True}


def evaluate_sample(dataset_entry: dict, strategy_name: str, raw_llm_response: str) -> dict:
    """
    dataset_entry: one entry from data/generated_dataset.json or
        data/dataset_with_clean.json
        (has: sample_id, error_type_id, error_line, buggy_code, ...;
         optional has_error, False for clean samples)
    strategy_name: e.g. "zero_shot"
    raw_llm_response: the exact text returned by LLMClient.generate()

    Returns a flat record ready to append to the results table.
    """
    parsed = parse_llm_response(raw_llm_response)
    parse_success = "_parse_error" not in parsed

    # Older dataset files contain buggy samples only and have no "has_error"
    # field, so a missing label means "buggy". Clean samples carry has_error=False.
    gt_has_error = bool(dataset_entry.get("has_error", True))

    record = {
        "sample_id": dataset_entry["sample_id"],
        "source_file": dataset_entry["source_file"],
        "error_type_id": dataset_entry["error_type_id"],
        "strategy": strategy_name,
        "ground_truth_has_error": gt_has_error,
        "ground_truth_line": dataset_entry["error_line"],
        "parse_success": parse_success,
        "raw_response": raw_llm_response,
    }

    if parse_success:
        record["predicted_has_error"] = parsed["has_error"]
        record["predicted_line"] = parsed["error_line"]
        record["predicted_description"] = parsed["error_description"]
        # Correct verdict = the model's has_error agrees with the label
        # (True for buggy samples, False for clean samples).
        record["correct_detection"] = (parsed["has_error"] == gt_has_error)
        # Localization is only defined for buggy samples that were flagged.
        record["correct_localization"] = bool(
            gt_has_error and parsed["has_error"]
            and parsed["error_line"] == dataset_entry["error_line"]
        )
    else:
        record["predicted_has_error"] = None
        record["predicted_line"] = None
        record["predicted_description"] = None
        record["correct_detection"] = False
        record["correct_localization"] = False

    return record
