"""Configuration for the guardrails.

A single dataclass holds every tunable, with sensible defaults. An
optional YAML file can override any of them — point ``GUARDRAIL_CONFIG``
at it, or pass the path to :func:`load_config`.

Example YAML override::

    pii_enabled: true
    injection_block_on_high: true
    grounding_pass_threshold: 0.7
    grounding_block_threshold: 0.3
    grounding_block_on_fail: false
    injection_extra_patterns:
      - name: internal_marker
        pattern: "TOP SECRET"
        severity: high
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GuardrailConfig:
    """All guardrail tunables in one place."""

    # -- PII redaction -------------------------------------------------
    pii_enabled: bool = True
    # "auto" (try Presidio, fall back to regex), "presidio", or "regex".
    # Also overridable with the GUARDRAIL_PII_BACKEND env var.
    pii_backend: str = "auto"
    # Placeholder format for redacted spans, e.g. "<EMAIL_ADDRESS>".
    pii_replacement_template: str = "<{entity_type}>"

    # -- Prompt-injection checks ---------------------------------------
    injection_enabled: bool = True
    # Block the request (HTTP 400) when a high-severity pattern matches.
    # Medium/low matches are reported but do not block.
    injection_block_on_high: bool = True
    # Extra patterns merged on top of the built-in list. Each item is a
    # mapping with keys: name, pattern (regex), severity (low/medium/high).
    injection_extra_patterns: list[dict[str, str]] = field(default_factory=list)
    # Names of built-in patterns to disable entirely.
    injection_disabled_patterns: list[str] = field(default_factory=list)

    # -- Grounding check -----------------------------------------------
    grounding_enabled: bool = True
    # Fraction of answer sentences that must be supported by the context
    # for a "pass" verdict.
    grounding_pass_threshold: float = 0.6
    # Below this fraction the verdict is "block"; between the two it is
    # "flag". Must be <= grounding_pass_threshold.
    grounding_block_threshold: float = 0.3
    # Minimum token overlap for one answer sentence to count as supported
    # by a single context chunk.
    grounding_sentence_threshold: float = 0.4
    # When True, the middleware rejects responses whose grounding verdict
    # is "block" instead of just annotating them.
    grounding_block_on_fail: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.grounding_block_threshold <= self.grounding_pass_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= grounding_block_threshold <= "
                "grounding_pass_threshold <= 1, got "
                f"{self.grounding_block_threshold=} {self.grounding_pass_threshold=}"
            )
        if not 0.0 <= self.grounding_sentence_threshold <= 1.0:
            raise ValueError(
                "grounding_sentence_threshold must be in [0, 1], got "
                f"{self.grounding_sentence_threshold}"
            )
        if self.pii_backend not in {"auto", "presidio", "regex"}:
            raise ValueError(
                f"pii_backend must be 'auto', 'presidio' or 'regex', "
                f"got {self.pii_backend!r}"
            )


def load_config(path: str | Path | None = None) -> GuardrailConfig:
    """Build a :class:`GuardrailConfig`, applying a YAML override if given.

    Resolution order: explicit ``path`` argument, then the
    ``GUARDRAIL_CONFIG`` environment variable, then defaults only.
    Unknown YAML keys raise :class:`ValueError` so a typo fails loudly
    instead of silently doing nothing.
    """
    resolved = path or os.environ.get("GUARDRAIL_CONFIG")
    overrides: dict[str, Any] = {}
    if resolved:
        import yaml  # lazy: PyYAML is small, but keep imports tidy

        with open(resolved, encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}

    known = set(GuardrailConfig.__dataclass_fields__)
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown guardrail config key(s): {sorted(unknown)}")

    # The env var wins over the YAML file for the PII backend, so a
    # deployment can force the lightweight regex path without editing files.
    env_backend = os.environ.get("GUARDRAIL_PII_BACKEND")
    if env_backend:
        overrides["pii_backend"] = env_backend.strip().lower()

    return GuardrailConfig(**overrides)
