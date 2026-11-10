"""
ground_truth_checker.py
------------------------
The authoritative, 100%-reliable source of truth for whether a code
snippet has a syntax error, using Python's own `ast` module — this is
what LLM responses get graded against.
"""

import ast


def check_syntax(code: str) -> dict:
    """
    Returns a dict:
        {
            "has_error": bool,
            "error_line": int or None,   # line number reported by Python itself
            "error_message": str or None
        }
    """
    try:
        ast.parse(code)
        return {"has_error": False, "error_line": None, "error_message": None}
    except SyntaxError as e:
        return {
            "has_error": True,
            "error_line": e.lineno,
            "error_message": str(e.msg),
        }


if __name__ == "__main__":
    good = "x = 1\nprint(x)"
    bad = "def f(x\n    return x"
    print("Clean code:", check_syntax(good))
    print("Buggy code:", check_syntax(bad))
