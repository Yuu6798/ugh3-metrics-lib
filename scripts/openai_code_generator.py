import argparse
import os
import subprocess
import sys
from typing import Optional

try:
    import openai
except ImportError:  # pragma: no cover - dependency may be missing
    openai = None


MODEL = "gpt-4o"
MAX_RETRIES = 3
OUTPUT_FILE = "ai_generated.py"


def generate_code(client: "openai.OpenAI", prompt: str) -> str:
    """Request Python code generation from OpenAI."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def run_generated_code() -> subprocess.CompletedProcess:
    """Run the generated Python file and return the completed process."""
    return subprocess.run(
        [sys.executable, OUTPUT_FILE],
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run Python code via OpenAI")
    parser.add_argument("text", help="Issue text prompting code generation")
    args = parser.parse_args()

    if openai is None:
        print("The 'openai' package is not installed.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    prompt = args.text
    code = generate_code(client, prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(code)
        result = run_generated_code()

        if result.returncode == 0:
            print(result.stdout)
            return

        if attempt == MAX_RETRIES:
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)

        error_info = result.stderr
        repair_prompt = (
            "The following Python code has an error. "
            "Please fix the error and output the full corrected code.\n\n" + code + "\n\nError:\n" + error_info
        )
        code = generate_code(client, repair_prompt)


if __name__ == "__main__":
    main()
