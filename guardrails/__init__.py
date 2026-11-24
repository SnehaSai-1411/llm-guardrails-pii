"""llm-guardrails-pii: standalone guardrail middleware for RAG/LLM services.

PII redaction, prompt-injection checks, and answer-grounding checks,
exposed both as plain functions and as FastAPI/Starlette middleware.

Heavy libraries (Presidio/spaCy) are imported lazily inside functions so
that importing this package — and collecting the test suite — stays light.
"""

from guardrails.config import GuardrailConfig, load_config

__all__ = ["GuardrailConfig", "load_config"]
