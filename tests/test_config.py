"""Config defaults, YAML overrides, and validation."""

import pytest

from guardrails.config import GuardrailConfig, load_config


def test_defaults():
    config = GuardrailConfig()
    assert config.pii_enabled is True
    assert config.pii_backend == "auto"
    assert config.injection_enabled is True
    assert config.injection_block_on_high is True
    assert config.grounding_enabled is True
    assert config.grounding_pass_threshold == 0.6
    assert config.grounding_block_threshold == 0.3
    assert config.grounding_block_on_fail is False


def test_yaml_override(tmp_path):
    yaml_file = tmp_path / "guardrail-config.yaml"
    yaml_file.write_text(
        "grounding_pass_threshold: 0.8\n"
        "grounding_block_on_fail: true\n"
        "injection_extra_patterns:\n"
        "  - name: custom_marker\n"
        "    pattern: 'BLUE FALCON'\n"
        "    severity: high\n"
    )
    config = load_config(yaml_file)
    assert config.grounding_pass_threshold == 0.8
    assert config.grounding_block_on_fail is True
    assert config.injection_extra_patterns[0]["name"] == "custom_marker"
    # Everything else keeps its default.
    assert config.pii_enabled is True


def test_unknown_yaml_key_raises(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("grounding_pass_threshhold: 0.8\n")
    with pytest.raises(ValueError, match="Unknown guardrail config key"):
        load_config(yaml_file)


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        GuardrailConfig(grounding_pass_threshold=0.2, grounding_block_threshold=0.5)
    with pytest.raises(ValueError):
        GuardrailConfig(grounding_sentence_threshold=1.5)


def test_invalid_pii_backend_raises():
    with pytest.raises(ValueError):
        GuardrailConfig(pii_backend="magic")
