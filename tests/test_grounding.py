"""Grounding-check tests."""

from guardrails import grounding
from guardrails.config import GuardrailConfig

CONTEXTS = [
    "Chroma is an open-source vector database for storing embeddings "
    "and running similarity search over them.",
    "Sentence-transformers runs locally to embed text, so no API key "
    "is needed to run the demo.",
]


def test_supported_answer_passes():
    answer = (
        "Chroma is an open-source vector database for storing embeddings "
        "and running similarity search over them."
    )
    result = grounding.check(answer, CONTEXTS)
    assert result["verdict"] == "pass"
    assert result["score"] == 1.0
    assert len(result["supported"]) == 1
    assert result["unsupported"] == []


def test_fabricated_claim_blocked():
    answer = "The Eiffel Tower was built in Paris in 1889 by Gustave Eiffel."
    result = grounding.check(answer, CONTEXTS)
    assert result["verdict"] == "block"
    assert result["score"] == 0.0
    assert len(result["unsupported"]) == 1


def test_half_supported_answer_flagged():
    answer = (
        "Chroma is an open-source vector database for storing embeddings "
        "and running similarity search over them. "
        "The Eiffel Tower was built in Paris in 1889 by Gustave Eiffel."
    )
    result = grounding.check(answer, CONTEXTS)
    assert result["score"] == 0.5
    assert result["verdict"] == "flag"
    assert len(result["supported"]) == 1
    assert len(result["unsupported"]) == 1


def test_no_context_blocks():
    answer = "Chroma is an open-source vector database."
    assert grounding.check(answer, [])["verdict"] == "block"
    assert grounding.check(answer, None)["verdict"] == "block"


def test_empty_answer_blocks():
    assert grounding.check("", CONTEXTS)["verdict"] == "block"
    assert grounding.check("   ", CONTEXTS)["score"] == 0.0


def test_single_string_context_accepted():
    result = grounding.check(
        "Chroma is an open-source vector database for storing embeddings.",
        CONTEXTS[0],
    )
    assert result["verdict"] == "pass"


def test_thresholds_are_configurable():
    # With a strict pass threshold, a half-supported answer no longer flags
    # as "flag" — it blocks (0.5 < 0.6 block threshold here).
    strict = GuardrailConfig(
        grounding_pass_threshold=0.9, grounding_block_threshold=0.6
    )
    answer = (
        "Chroma is an open-source vector database for storing embeddings "
        "and running similarity search over them. "
        "The Eiffel Tower was built in Paris in 1889 by Gustave Eiffel."
    )
    assert grounding.check(answer, CONTEXTS, strict)["verdict"] == "block"
