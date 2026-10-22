import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

# api.py loads the embedding/reranker models at import, so load only its
# generation code, the same way the eval does.
_spec = importlib.util.spec_from_file_location(
    "evaluate_retrieval", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "evaluate_retrieval.py"))
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
llm = _eval.load_from_api()


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(response):
    models = FakeModels(response)
    return SimpleNamespace(models=models), models


def gemini_response(text="", block_reason=None, finish_reason="STOP"):
    return SimpleNamespace(
        text=text,
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name=block_reason)) if block_reason else None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))] if finish_reason else [],
    )


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_stripped_text_and_sends_prompt(self):
        client, models = fake_client(gemini_response("  ## Answer\nThirty days.\n"))
        self.assertEqual(llm.generate_answer(client, "prompt"), "## Answer\nThirty days.")
        self.assertEqual(models.kwargs["contents"], "prompt")
        self.assertEqual(models.kwargs["model"], llm.get_llm_model())

    def test_empty_text_returns_empty_string(self):
        client, _ = fake_client(gemini_response(text=None))
        self.assertEqual(llm.generate_answer(client, "prompt"), "")

    def test_blocked_prompt_raises_refusal(self):
        client, _ = fake_client(gemini_response(block_reason="SAFETY", finish_reason=None))
        with self.assertRaises(llm.RefusalError) as ctx:
            llm.generate_answer(client, "prompt")
        self.assertEqual(ctx.exception.category, "SAFETY")

    def test_blocked_answer_raises_refusal(self):
        client, _ = fake_client(gemini_response(finish_reason="RECITATION"))
        with self.assertRaises(llm.RefusalError) as ctx:
            llm.generate_answer(client, "prompt")
        self.assertEqual(ctx.exception.category, "RECITATION")

    def test_model_from_env_with_default(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL": ""}):
            self.assertEqual(llm.get_llm_model(), "gemini-2.5-flash")
        with mock.patch.dict(os.environ, {"LLM_MODEL": "gemini-2.5-pro"}):
            self.assertEqual(llm.get_llm_model(), "gemini-2.5-pro")

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                llm.make_llm_client()


if __name__ == "__main__":
    unittest.main()
