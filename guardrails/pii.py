"""PII detection and redaction.

Primary path: Microsoft Presidio (``presidio-analyzer`` +
``presidio-anonymizer``) backed by the small spaCy model
``en_core_web_sm``. Presidio is imported lazily — only when first used —
so importing this module never pulls in spaCy.

Fallback path: if Presidio or the spaCy model is unavailable (or the
``GUARDRAIL_PII_BACKEND=regex`` env var / ``pii_backend="regex"`` config
forces it), a small regex recognizer set covers email, phone, SSN,
credit card, and US street address. The public API is identical either
way; :func:`redact` reports which backend ran in the ``backend`` field,
and :func:`get_backend` tells you which path is active without redacting
anything.

Do not treat the regex fallback as equivalent to Presidio: it catches
well-formatted identifiers only, misses names/locations, and can miss
unusual formats. It exists so the guardrail still does *something*
useful in a lightweight environment, and so the test suite runs offline
without the spaCy model.
"""

from __future__ import annotations

import os
import re
from typing import Any

from guardrails.config import GuardrailConfig

# ---------------------------------------------------------------------------
# Regex fallback recognizers
# ---------------------------------------------------------------------------
# Order matters: more specific patterns (SSN, credit card) run before the
# generic phone pattern so a SSN is not half-matched as a phone number.

_REGEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "US_SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        # 13-16 digits with optional spaces/dashes, validated by Luhn below.
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    ),
    (
        "PHONE_NUMBER",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
        ),
    ),
    (
        # US street address, e.g. "1265 E University Dr" / "221B Baker Street".
        # Deliberately requires a street suffix so "Room 12" is not flagged.
        "US_ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s+){0,3}?"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|"
            r"Drive|Dr|Court|Ct|Way|Place|Pl)\b",
            re.IGNORECASE,
        ),
    ),
]


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum — weeds out random 16-digit runs that aren't cards."""
    nums = [int(d) for d in digits]
    if not 13 <= len(nums) <= 16:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _regex_redact(text: str, template: str = "<{entity_type}>") -> dict[str, Any]:
    """Redact with the regex recognizer set. Always available, no deps."""
    # Collect non-overlapping matches, first pattern wins on overlap.
    found: list[dict[str, Any]] = []
    claimed: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in claimed)

    for entity_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if _overlaps(start, end):
                continue
            if entity_type == "CREDIT_CARD":
                digits = re.sub(r"\D", "", match.group())
                if not _luhn_valid(digits):
                    continue
            claimed.append((start, end))
            found.append(
                {
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "score": 0.85,  # regex confidence: high, not certain
                }
            )

    found.sort(key=lambda e: e["start"])
    redacted = text
    # Replace from the end so earlier offsets stay valid.
    for entity in reversed(found):
        replacement = template.format(entity_type=entity["entity_type"])
        redacted = (
            redacted[: entity["start"]] + replacement + redacted[entity["end"] :]
        )

    return {
        "redacted_text": redacted,
        "entities_found": found,
        "backend": "regex",
    }


# ---------------------------------------------------------------------------
# Presidio path (lazy)
# ---------------------------------------------------------------------------

_presidio_engines: tuple[Any, Any] | None = None
_presidio_failed: bool = False


def _load_presidio() -> tuple[Any, Any] | None:
    """Build (analyzer, anonymizer) once; return None if unavailable.

    Any failure — package missing, spaCy model missing, model load error —
    is cached and the caller falls back to regex. Importing here (not at
    module top) keeps pytest collection light.
    """
    global _presidio_engines, _presidio_failed
    if _presidio_engines is not None:
        return _presidio_engines
    if _presidio_failed:
        return None
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        # Pin the SMALL spaCy model explicitly. Presidio's default config
        # wants en_core_web_lg (~560 MB) and will try to pip-install it on
        # the spot if it is missing — far too heavy for this project.
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        anonymizer = AnonymizerEngine()
    except Exception:  # noqa: BLE001 — any load failure means "no Presidio"
        _presidio_failed = True
        return None
    _presidio_engines = (analyzer, anonymizer)
    return _presidio_engines


def presidio_available() -> bool:
    """True if Presidio + the spaCy model load successfully in this env."""
    return _load_presidio() is not None


def _presidio_redact(text: str, template: str = "<{entity_type}>") -> dict[str, Any]:
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _load_presidio()  # type: ignore[misc]
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return {"redacted_text": text, "entities_found": [], "backend": "presidio"}
    # One replace operator per entity type so the placeholder names the type.
    operators = {
        r.entity_type: OperatorConfig(
            "replace", {"new_value": template.format(entity_type=r.entity_type)}
        )
        for r in results
    }
    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    )
    return {
        "redacted_text": anonymized.text,
        "entities_found": [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
            }
            for r in results
        ],
        "backend": "presidio",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_backend(config: GuardrailConfig | None = None) -> str:
    """Which PII backend would :func:`redact` use: "presidio" or "regex"."""
    cfg = config or GuardrailConfig()
    backend = os.environ.get("GUARDRAIL_PII_BACKEND", cfg.pii_backend).strip().lower()
    if backend == "regex":
        return "regex"
    if backend == "presidio":
        return "presidio" if presidio_available() else "regex"
    return "presidio" if presidio_available() else "regex"


def redact(text: str, config: GuardrailConfig | None = None) -> dict[str, Any]:
    """Redact PII from ``text``.

    Returns ``{"redacted_text": str, "entities_found": [...],
    "backend": "presidio" | "regex"}``. ``entities_found`` entries carry
    ``entity_type``, ``start``, ``end`` and ``score``. Empty/whitespace
    input passes through untouched.
    """
    cfg = config or GuardrailConfig()
    if not cfg.pii_enabled or not text or not text.strip():
        return {"redacted_text": text, "entities_found": [], "backend": "disabled" if not cfg.pii_enabled else get_backend(cfg)}
    backend = get_backend(cfg)
    if backend == "presidio":
        try:
            return _presidio_redact(text, cfg.pii_replacement_template)
        except Exception:  # noqa: BLE001 — degrade, never crash the request
            return _regex_redact(text, cfg.pii_replacement_template)
    return _regex_redact(text, cfg.pii_replacement_template)
