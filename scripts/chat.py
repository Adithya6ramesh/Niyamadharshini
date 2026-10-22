"""
Terminal chat client (Malayalam + English).

Uses the same retrieval, prompt and Claude model as the web app by importing
them from api.py (which loads the models on import), so both stay identical.

    python scripts/chat.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from api import RefusalError, build_prompt, generate_answer, llm_client, retrieve_context


def ask(question: str):
    print("\n[SEARCH] Retrieving relevant documents...")
    contexts, sources = retrieve_context(question)

    if not contexts:
        print("ബന്ധപ്പെട്ട രേഖകൾ കണ്ടെത്താനായില്ല.")
        return

    print("[LLM] Generating answer...")
    try:
        answer = generate_answer(llm_client, build_prompt(question, contexts))
    except RefusalError as refusal:
        print(f"This question could not be answered ({refusal}).")
        return
    except anthropic.APIError as error:
        print(f"AI service temporarily unavailable: {error}")
        return

    print("\n" + "=" * 80 + "\nANSWER\n" + "=" * 80)
    print(answer or "ലഭ്യമായ രേഖകളിൽ നിന്ന് വ്യക്തമായ നിയമപരമായ വിശദീകരണം ലഭ്യമല്ല.")

    print("\n" + "=" * 80 + "\nSOURCES\n" + "=" * 80)
    for s in sources:
        print(f"- {s['pdf']} | Page {s['page']} | relevance={s['relevance']}")


if __name__ == "__main__":
    print("\nMultilingual RAG Chat (type 'exit' to quit)\n")
    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            ask(query)
