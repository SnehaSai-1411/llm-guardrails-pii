"""Output grounding check.

Given an answer and the retrieved context chunks it was supposed to be
drawn from, score how much of the answer is actually supported by the
context — sentence by sentence, by token overlap. This is an
"entailment-lite" heuristic, not a real NLI model: cheap, offline,
deterministic, and honest about what it measures. It catches fabricated
claims that introduce vocabulary absent from the context; it will not
catch a fluent paraphrase that contradicts the context, or a claim that
reuses context words in a false combination.

Verdicts, from the config thresholds::

    score >= pass_threshold   -> "pass"
    score >= block_threshold  -> "flag"
    otherwise                 -> "block"

where ``score`` is the fraction of answer sentences substantially
supported by at least one context chunk.
"""

from __future__ import annotations

import re
from typing import Any

from guardrails.config import GuardrailConfig

# Small stopword set — enough that "the answer is" does not count as
# support, without pulling in an NLP dependency for this module.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on at for with by
    from as is are was were be been being am do does did doing have has had
    having i you he she it we they me him her us them my your his its our
    their this that these those there here what which who whom how why when
    where can could should would may might must shall will just not no yes
    so very more most less least also than then too
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts


def _content_tokens(text: str) -> set[str]:
    tokens = set(_TOKEN.findall(text.lower()))
    content = tokens - _STOPWORDS
    # Very short sentences ("Yes.") would be empty after stopword removal;
    # fall back to all tokens so they can still match.
    return content or tokens


def _sentence_support(sentence: str, contexts: list[str]) -> float:
    """Best token-overlap of one sentence against any context chunk."""
    answer_tokens = _content_tokens(sentence)
    if not answer_tokens:
        return 0.0
    best = 0.0
    for chunk in contexts:
        chunk_tokens = _content_tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
        best = max(best, overlap)
    return best


def check(
    answer: str,
    contexts: list[str] | str | None,
    config: GuardrailConfig | None = None,
) -> dict[str, Any]:
    """Check whether ``answer`` is grounded in ``contexts``.

    Returns ``{"score": float, "supported": [...], "unsupported": [...],
    "verdict": "pass"|"flag"|"block"}``. ``supported``/``unsupported``
    list the answer sentences in each bucket; ``score`` is the supported
    fraction. An empty answer, or no context at all, scores 0 ("block"):
    there is nothing to ground the answer in.
    """
    cfg = config or GuardrailConfig()
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts = [c for c in (contexts or []) if c and c.strip()]

    sentences = _sentences(answer) if answer and answer.strip() else []
    if not sentences or not contexts:
        return {
            "score": 0.0,
            "supported": [],
            "unsupported": sentences,
            "verdict": "block",
        }

    supported: list[str] = []
    unsupported: list[str] = []
    for sentence in sentences:
        if _sentence_support(sentence, contexts) >= cfg.grounding_sentence_threshold:
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    score = len(supported) / len(sentences)
    if score >= cfg.grounding_pass_threshold:
        verdict = "pass"
    elif score >= cfg.grounding_block_threshold:
        verdict = "flag"
    else:
        verdict = "block"
    return {
        "score": round(score, 3),
        "supported": supported,
        "unsupported": unsupported,
        "verdict": verdict,
    }
