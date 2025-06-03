import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import ai_issue_codegen
import inspect


def test_default_model_is_gpt4o() -> None:
    """Ensure llm uses gpt-4o by default."""
    signature = inspect.signature(ai_issue_codegen.llm)
    model_param = signature.parameters["model"]
    assert model_param.default == "gpt-4o"
