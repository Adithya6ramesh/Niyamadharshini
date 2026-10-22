import os
import unittest
from types import SimpleNamespace
from unittest import mock

import importlib.util

# api.py loads the embedding/reranker models at import, so load only its
# generation code, the same way the eval does.
_spec = importlib.util.spec_from_file_location(
    "evaluate_retrieval", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "evaluate_retrieval.py"))
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
llm = _eval.load_from_api()


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


class GenerateAnswerTests(unittest.TestCase):
    def test_joins_text_blocks_and_skips_thinking(self):
        response = SimpleNamespace(stop_reason="end_turn", stop_details=None, content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="## Answer\n"),
            SimpleNamespace(type="text", text="Thirty days."),
        ])
        client, messages = fake_client(response)
        self.assertEqual(llm.generate_answer(client, "prompt"), "## Answer\nThirty days.")
        self.assertEqual(messages.kwargs["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(messages.kwargs["fallbacks"], "default")
        self.assertEqual(messages.kwargs["betas"], [llm.LLM_FALLBACK_BETA])

    def test_refusal_raises_with_category(self):
        response = SimpleNamespace(stop_reason="refusal", content=[],
                                   stop_details=SimpleNamespace(category="general_harms"))
        client, _ = fake_client(response)
        with self.assertRaises(llm.RefusalError) as ctx:
            llm.generate_answer(client, "prompt")
        self.assertEqual(ctx.exception.category, "general_harms")

    def test_model_from_env_with_default(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL": ""}):
            self.assertEqual(llm.get_llm_model(), "claude-sonnet-5-5")
        with mock.patch.dict(os.environ, {"LLM_MODEL": "claude-opus-5-5"}):
            self.assertEqual(llm.get_llm_model(), "claude-opus-5-5")

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                llm.make_llm_client()


if __name__ == "__main__":
    unittest.main()
