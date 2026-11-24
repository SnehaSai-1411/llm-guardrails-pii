"""Demo app: a tiny FastAPI service with the guardrails mounted.

The ``/ask`` endpoint is fake on purpose — it answers from a canned
context about this repo's own stack, so the guardrails can be exercised
end to end with curl and no LLM key:

    uvicorn demo_app:app --reload
    curl -X POST localhost:8000/ask \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "My email is ada@example.com, what is a vector database?"}'

The question that reaches the endpoint has the email redacted by the
middleware, and the response carries a ``grounding`` verdict computed
against the canned context.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from guardrails.middleware import GuardrailMiddleware

CANNED_CONTEXT = [
    "Chroma is an open-source vector database for storing embeddings "
    "and running similarity search over them.",
    "This demo uses sentence-transformers locally to embed text, so no "
    "API key is needed to run it.",
]

CANNED_ANSWER = (
    "Chroma is an open-source vector database for storing embeddings "
    "and running similarity search over them."
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    contexts: list[str]
    question_seen_by_endpoint: str


app = FastAPI(title="llm-guardrails-pii demo")
app.add_middleware(GuardrailMiddleware, grounding_paths=["/ask"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    # Fake retrieval + generation: the canned answer is drawn verbatim from
    # the canned context, so the grounding check passes.
    return AskResponse(
        answer=CANNED_ANSWER,
        contexts=CANNED_CONTEXT,
        question_seen_by_endpoint=body.question,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo_app:app", host="127.0.0.1", port=8000, reload=True)
