"""Prompt-injection checks.

A configurable list of regex patterns over common injection phrasings,
each with a severity (low / medium / high). The middleware blocks on
high-severity matches only; everything else is reported so the caller
can decide.

The patterns are deliberately anchored to *instruction-override*
phrasings ("ignore previous instructions", "disregard your system
prompt") rather than to bare verbs like "ignore" — a benign request such
as "ignore the typo in my question" must not flag. See
``tests/test_injection.py`` for the false-positive guard tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from guardrails.config import GuardrailConfig

SEVERITIES = ("low", "medium", "high")
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class InjectionPattern:
    """One named detection pattern."""

    name: str
    pattern: str
    severity: str  # low | medium | high
    description: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        re.compile(self.pattern)  # fail fast on a bad regex

    def search(self, text: str) -> re.Match[str] | None:
        return re.search(self.pattern, text, re.IGNORECASE)


# Built-in patterns. Keep them specific: a pattern that fires on ordinary
# user language is worse than a pattern that misses an exotic attack.
DEFAULT_PATTERNS: list[InjectionPattern] = [
    InjectionPattern(
        name="ignore_previous_instructions",
        pattern=r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|directions?)\b",
        severity="high",
        description="Direct attempt to override earlier instructions.",
    ),
    InjectionPattern(
        name="disregard_system_prompt",
        pattern=r"\b(?:disregard|ignore|override|forget|bypass)\b.{0,40}?"
        r"\b(?:system\s+prompt|system\s+instructions?|safety\s+"
        r"(?:guidelines|rules|instructions?)|content\s+policy)\b",
        severity="high",
        description="Attempt to discard the system prompt or safety rules.",
    ),
    InjectionPattern(
        name="you_are_now",
        pattern=r"\byou\s+are\s+now\b.{0,60}?\b(?:a|an|the)?\s*"
        r"(?:\w+\s+){0,3}?(?:without|with\s+no|no)\s+"
        r"(?:restrictions?|limits?|rules?|filters?|safety)\b"
        r"|\byou\s+are\s+now\s+(?:in\s+)?(?:developer|admin|god|jailbreak|"
        r"unrestricted|dan)\s+mode\b",
        severity="high",
        description="Role-reassignment jailbreak ('you are now ...').",
    ),
    InjectionPattern(
        name="forget_everything",
        pattern=r"\bforget\s+(?:everything|all)\b.{0,30}?\b"
        r"(?:you\s+(?:know|were\s+told)|your\s+(?:instructions?|training|rules?))\b",
        severity="high",
        description="Attempt to wipe the model's instructions.",
    ),
    InjectionPattern(
        name="pretend_no_restrictions",
        pattern=r"\bpretend\s+(?:you\s+)?(?:have\s+no|there\s+are\s+no)\s+"
        r"(?:restrictions?|rules?|limits?|filters?)\b"
        r"|\bact\s+as\s+if\s+you\s+have\s+no\s+"
        r"(?:restrictions?|rules?|limits?|filters?)\b",
        severity="medium",
        description="Roleplay framing used to dodge restrictions.",
    ),
    InjectionPattern(
        name="delimiter_escape",
        pattern=r"(?:\[/?INST\]|<<\s*SYS\s*>>|</?\s*system\s*>|"
        r"---\s*END\s+(?:SYSTEM|INSTRUCTIONS?)\s*---|"
        r"###\s*(?:SYSTEM|INSTRUCTION|NEW\s+INSTRUCTIONS?)\s*:)",
        severity="medium",
        description="Prompt-template delimiters pasted into user input.",
    ),
    InjectionPattern(
        name="reveal_system_prompt",
        pattern=r"\b(?:reveal|show|print|repeat|output|display)\b.{0,30}?"
        r"\b(?:your\s+)?(?:system\s+prompt|initial\s+(?:prompt|instructions?)|"
        r"hidden\s+instructions?)\b",
        severity="medium",
        description="Attempt to exfiltrate the system prompt.",
    ),
    InjectionPattern(
        name="do_anything_now",
        pattern=r"\bdo\s+anything\s+now\b|\bDAN\b.{0,20}?\bmode\b",
        severity="medium",
        description="Known 'Do Anything Now' jailbreak marker.",
    ),
    InjectionPattern(
        name="instruction_in_data",
        pattern=r"\b(?:new\s+)?instructions?\s*:\s*(?:ignore|disregard|do\s+not\s+follow)\b",
        severity="low",
        description="Instruction-like text embedded in pasted content.",
    ),
]


def get_patterns(config: GuardrailConfig | None = None) -> list[InjectionPattern]:
    """Built-in patterns, minus disabled ones, plus YAML/config extras."""
    cfg = config or GuardrailConfig()
    disabled = set(cfg.injection_disabled_patterns)
    patterns = [p for p in DEFAULT_PATTERNS if p.name not in disabled]
    for extra in cfg.injection_extra_patterns:
        patterns.append(
            InjectionPattern(
                name=extra["name"],
                pattern=extra["pattern"],
                severity=extra.get("severity", "medium"),
                description=extra.get("description", ""),
            )
        )
    return patterns


def check(text: str, config: GuardrailConfig | None = None) -> dict[str, Any]:
    """Check ``text`` for prompt-injection patterns.

    Returns ``{"flagged": bool, "matches": [...],
    "highest_severity": "low"|"medium"|"high"|None}``. Each match carries
    ``pattern`` (name), ``severity``, ``matched_text`` and ``description``.
    ``flagged`` is True when at least one pattern matches; blocking policy
    (high severity only) lives in the middleware, not here.
    """
    cfg = config or GuardrailConfig()
    if not cfg.injection_enabled or not text or not text.strip():
        return {"flagged": False, "matches": [], "highest_severity": None}
    matches: list[dict[str, Any]] = []
    for pattern in get_patterns(cfg):
        found = pattern.search(text)
        if found:
            matches.append(
                {
                    "pattern": pattern.name,
                    "severity": pattern.severity,
                    "matched_text": found.group(0),
                    "description": pattern.description,
                }
            )
    highest = None
    if matches:
        highest = max(matches, key=lambda m: _SEVERITY_RANK[m["severity"]])["severity"]
    return {"flagged": bool(matches), "matches": matches, "highest_severity": highest}


def is_blocked(result: dict[str, Any], config: GuardrailConfig | None = None) -> bool:
    """Should this check result block the request? High severity only."""
    cfg = config or GuardrailConfig()
    return bool(
        cfg.injection_block_on_high
        and result.get("flagged")
        and result.get("highest_severity") == "high"
    )
