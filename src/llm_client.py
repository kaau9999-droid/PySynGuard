"""
llm_client.py
-------------
A thin, provider-agnostic wrapper around LLM APIs so the rest of the
pipeline (prompting_strategies.py, evaluator.py, experiments) never has
to know which provider is actually being called.

Supported providers (choose one via the PYSYNGUARD_PROVIDER env var):
    - "gemini"  -> uses the google-genai SDK (has a generous free tier)
    - "openai"  -> uses openai python SDK
    - "anthropic" -> uses anthropic python SDK
    - "groq"    -> uses openai SDK pointed at Groq's OpenAI-compatible endpoint
                   (very generous free tier + extremely fast inference — good
                   backup when another provider's quota runs out)

Set the relevant API key as an environment variable before running:
    GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY

There is also a "mock" provider (PYSYNGUARD_PROVIDER=mock) that requires
NO API key and NO internet access. It simulates an imperfect LLM so you
can test the entire pipeline (dataset -> prompting -> evaluation -> UI)
end-to-end for free. IMPORTANT: swap to a real provider before generating
the final numbers you put in the report — the mock numbers are synthetic
and only meant to validate that the pipeline works.

Usage:
    from llm_client import LLMClient
    client = LLMClient()  # reads provider + model from env, with sane defaults
    response_text = client.generate("your prompt here")
"""

import json
import os
import random
import re
import time

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the project root, if present
except ImportError:
    pass  # python-dotenv not installed; fall back to real env vars only


class LLMClient:
    def __init__(self, provider: str = None, model: str = None, temperature: float = 0.0):
        self.provider = provider or os.getenv("PYSYNGUARD_PROVIDER", "gemini")
        self.temperature = temperature

        if self.provider == "gemini":
            self.model = model or os.getenv("PYSYNGUARD_MODEL", "gemini-2.5-flash")
            self._init_gemini()
        elif self.provider == "openai":
            self.model = model or os.getenv("PYSYNGUARD_MODEL", "gpt-4o-mini")
            self._init_openai()
        elif self.provider == "anthropic":
            self.model = model or os.getenv("PYSYNGUARD_MODEL", "claude-sonnet-4-6")
            self._init_anthropic()
        elif self.provider == "groq":
            # openai/gpt-oss-120b is Groq's current recommended general-purpose
            # model (llama-3.3-70b-versatile was deprecated in 2026).
            self.model = model or os.getenv("PYSYNGUARD_MODEL", "openai/gpt-oss-120b")
            self._init_groq()
        elif self.provider == "mock":
            self.model = "mock-llm-v1"
            # no init needed
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ---- Provider initializers ----
    def _init_gemini(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def _init_openai(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)

    def _init_anthropic(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _init_groq(self):
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    # ---- Unified generate() ----
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        """Send a prompt, return the raw text response. Retries on transient errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    from google.genai import types
                    response = self._client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=self.temperature),
                    )
                    return response.text.strip()

                elif self.provider == "openai":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "groq":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "anthropic":
                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=1000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text.strip()

                elif self.provider == "mock":
                    return self._mock_generate(prompt)

            except Exception as e:
                last_error = e
                time.sleep(retry_delay * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    # ---- Mock provider internals ----
    # Simulates a plausible LLM with strategy-dependent accuracy, computed
    # against the REAL ground truth (via ast) extracted from the prompt's
    # line-numbered code block. This exists only to let the pipeline be
    # tested for free/offline. Replace with a real provider for report numbers.
    _STRATEGY_ACCURACY = {
        "step by step": 0.85,       # cot_prompt signature phrase
        "worked examples": 0.80,    # few_shot_prompt signature phrase
        "strict Python interpreter": 0.75,  # role_based_prompt signature phrase
        "_default": 0.65,           # zero_shot_prompt (no special phrase)
    }

    def _mock_generate(self, prompt: str) -> str:
        import ast as _ast

        # Recover the original code from the "N: <line>" formatted block that
        # appears AFTER the CODE_UNDER_TEST_MARKER, so few-shot examples
        # (which are also numbered) are never mistaken for the real snippet.
        marker = "### CODE UNDER TEST ###"
        section = prompt.split(marker, 1)[-1] if marker in prompt else prompt

        code_lines = []
        for line in section.splitlines():
            m = re.match(r"^\d+:\s?(.*)$", line)
            if m:
                code_lines.append(m.group(1))
            elif code_lines:
                # Stop at the first non-numbered line after we've started
                # collecting the snippet (e.g. the blank line + next instructions).
                break
        code = "\n".join(code_lines)

        try:
            _ast.parse(code)
            true_has_error, true_line = False, None
        except SyntaxError as e:
            true_has_error, true_line = True, e.lineno

        # Pick simulated accuracy based on which strategy phrase appears in the prompt.
        accuracy = self._STRATEGY_ACCURACY["_default"]
        for phrase, acc in self._STRATEGY_ACCURACY.items():
            if phrase != "_default" and phrase in prompt:
                accuracy = acc
                break

        correct = random.random() < accuracy

        if correct:
            has_error, error_line = true_has_error, true_line
            desc = "Simulated correct detection" if has_error else ""
        else:
            # Simulate a plausible mistake: flip the has_error verdict,
            # or (if there IS an error) report a nearby wrong line.
            if true_has_error and random.random() < 0.5 and code_lines:
                has_error = True
                error_line = max(1, true_line + random.choice([-1, 1]))
                desc = "Simulated incorrect line localization"
            else:
                has_error = not true_has_error
                error_line = None if not has_error else (true_line or 1)
                desc = "Simulated incorrect verdict"

        return json.dumps({
            "has_error": has_error,
            "error_line": error_line,
            "error_description": desc,
        })


if __name__ == "__main__":
    # Quick manual smoke test:
    #   PYSYNGUARD_PROVIDER=gemini GEMINI_API_KEY=xxx python3 llm_client.py
    client = LLMClient()
    print(client.generate("Say 'PySynGuard client is working' and nothing else."))
