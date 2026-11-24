"""PII redaction tests.

The regex-fallback tests always run (forced via config, no spaCy needed).
The Presidio tests skip gracefully when Presidio or the spaCy model is
unavailable in the environment.
"""

import pytest

from guardrails import pii
from guardrails.config import GuardrailConfig

REGEX_CONFIG = GuardrailConfig(pii_backend="regex")
PRESIDIO_CONFIG = GuardrailConfig(pii_backend="presidio")


def test_email_redacted_regex():
    result = pii.redact("Contact me at ada@example.com please.", REGEX_CONFIG)
    assert "ada@example.com" not in result["redacted_text"]
    assert "<EMAIL_ADDRESS>" in result["redacted_text"]
    assert result["entities_found"][0]["entity_type"] == "EMAIL_ADDRESS"
    assert result["backend"] == "regex"


def test_phone_redacted_regex():
    result = pii.redact("Call me at (555) 123-4567 tomorrow.", REGEX_CONFIG)
    assert "(555) 123-4567" not in result["redacted_text"]
    assert "<PHONE_NUMBER>" in result["redacted_text"]


def test_ssn_redacted_regex():
    result = pii.redact("My SSN is 123-45-6789.", REGEX_CONFIG)
    assert "123-45-6789" not in result["redacted_text"]
    assert "<US_SSN>" in result["redacted_text"]


def test_credit_card_redacted_regex():
    # 4242... is the standard Stripe test card number (valid Luhn).
    result = pii.redact("Card: 4242 4242 4242 4242.", REGEX_CONFIG)
    assert "4242 4242 4242 4242" not in result["redacted_text"]
    assert "<CREDIT_CARD>" in result["redacted_text"]


def test_random_long_number_not_a_card():
    # 16 digits that fail the Luhn checksum must not be treated as a card.
    result = pii.redact("Order ref 1111111111111111 shipped.", REGEX_CONFIG)
    assert "<CREDIT_CARD>" not in result["redacted_text"]
    assert "1111111111111111" in result["redacted_text"]


def test_us_address_redacted_regex():
    result = pii.redact("I live at 221 Baker Street.", REGEX_CONFIG)
    assert "221 Baker Street" not in result["redacted_text"]
    assert "<US_ADDRESS>" in result["redacted_text"]


def test_normal_text_untouched():
    text = "The quick brown fox jumps over the lazy dog."
    result = pii.redact(text, REGEX_CONFIG)
    assert result["redacted_text"] == text
    assert result["entities_found"] == []


def test_empty_text_passes_through():
    assert pii.redact("", REGEX_CONFIG)["redacted_text"] == ""
    assert pii.redact("   ", REGEX_CONFIG)["redacted_text"] == "   "


def test_multiple_entities_in_one_text():
    result = pii.redact(
        "Ada Lovelace, ada@example.com, phone (555) 123-4567.", REGEX_CONFIG
    )
    types = {e["entity_type"] for e in result["entities_found"]}
    assert types == {"EMAIL_ADDRESS", "PHONE_NUMBER"}
    assert "ada@example.com" not in result["redacted_text"]
    assert "(555) 123-4567" not in result["redacted_text"]


def test_pii_disabled_passes_through():
    config = GuardrailConfig(pii_enabled=False, pii_backend="regex")
    result = pii.redact("ada@example.com", config)
    assert result["redacted_text"] == "ada@example.com"
    assert result["entities_found"] == []


@pytest.mark.skipif(
    not pii.presidio_available(),
    reason="Presidio or the en_core_web_sm spaCy model is not installed",
)
def test_presidio_path_redacts_email():
    result = pii.redact("Contact me at ada@example.com please.", PRESIDIO_CONFIG)
    assert result["backend"] == "presidio"
    assert "ada@example.com" not in result["redacted_text"]
    assert "<EMAIL_ADDRESS>" in result["redacted_text"]


@pytest.mark.skipif(
    not pii.presidio_available(),
    reason="Presidio or the en_core_web_sm spaCy model is not installed",
)
def test_presidio_path_detects_person_name():
    # The regex fallback cannot do names; Presidio should.
    result = pii.redact("My name is Ada Lovelace.", PRESIDIO_CONFIG)
    assert result["backend"] == "presidio"
    types = {e["entity_type"] for e in result["entities_found"]}
    assert "PERSON" in types
