"""
prompting_strategies.py
------------------------
Defines the 4 prompting strategies compared in the experiment. Every
strategy asks the LLM to return a STRICT JSON object so that responses
can be parsed programmatically and compared against ground truth:

    {
        "has_error": true/false,
        "error_line": <int or null>,
        "error_description": "<short string>"
    }

Each function takes the code under test (and, for few-shot, a couple of
worked examples) and returns the final prompt string to send to the LLM.
"""

CODE_UNDER_TEST_MARKER = "### CODE UNDER TEST ###"

RESPONSE_FORMAT_INSTRUCTIONS = """
Respond with ONLY a valid JSON object in exactly this format, and nothing else
(no markdown fences, no extra commentary):

{
  "has_error": true or false,
  "error_line": <line number as an integer, or null if no error>,
  "error_description": "<one short sentence describing the error, or empty string>"
}
""".strip()


def zero_shot_prompt(code: str) -> str:
    return f"""You are analyzing a Python code snippet for syntax errors.

Code (line numbers added for reference):
### CODE UNDER TEST ###
{_add_line_numbers(code)}

Does this code contain a syntax error? If so, identify the line number.

{RESPONSE_FORMAT_INSTRUCTIONS}
"""


def few_shot_prompt(code: str, examples: list = None) -> str:
    if examples is None:
        examples = _default_few_shot_examples()

    examples_text = "\n\n".join(
        f"Example {i+1}:\nCode:\n{_add_line_numbers(ex['code'])}\nAnswer:\n{ex['answer']}"
        for i, ex in enumerate(examples)
    )

    return f"""You are analyzing Python code snippets for syntax errors.
Here are some worked examples:

{examples_text}

Now analyze this new snippet:
Code (line numbers added for reference):
### CODE UNDER TEST ###
{_add_line_numbers(code)}

Does this code contain a syntax error? If so, identify the line number.

{RESPONSE_FORMAT_INSTRUCTIONS}
"""


def cot_prompt(code: str) -> str:
    return f"""You are analyzing a Python code snippet for syntax errors.

Code (line numbers added for reference):
### CODE UNDER TEST ###
{_add_line_numbers(code)}

Think step by step: examine each line one at a time, checking for missing colons,
unmatched brackets/parentheses/quotes, bad indentation, missing operators, and
invalid syntax. Reason through your analysis internally, then give your final answer.

{RESPONSE_FORMAT_INSTRUCTIONS}
"""


def role_based_prompt(code: str) -> str:
    return f"""You are a strict Python interpreter performing static syntax analysis,
equivalent to running `ast.parse()` on the source code. You do not execute the code,
you only check whether it is syntactically well-formed Python.

Code (line numbers added for reference):
### CODE UNDER TEST ###
{_add_line_numbers(code)}

As the interpreter, report whether parsing this code would raise a SyntaxError,
and if so, on which line.

{RESPONSE_FORMAT_INSTRUCTIONS}
"""


# ---- Helpers ----

def _add_line_numbers(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))


def _default_few_shot_examples():
    """Two small, hand-written examples (NOT drawn from the actual dataset,
    to avoid leaking test samples into the prompt)."""
    return [
        {
            "code": "def greet(name)\n    print(name)",
            "answer": '{"has_error": true, "error_line": 1, "error_description": "Missing colon after function definition"}',
        },
        {
            "code": "def greet(name):\n    print(name)\n\ngreet(\"Ali\")",
            "answer": '{"has_error": false, "error_line": null, "error_description": ""}',
        },
    ]


STRATEGIES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "role_based": role_based_prompt,
}
