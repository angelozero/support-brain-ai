"""Tests for the NovaMind support agent prompt."""

import pytest

from prompts import PROMPT_VIRTUAL_ASSISTANCE


class TestPromptVirtualAssistance:
    """Validate the system prompt structure and content."""

    def test_prompt_is_non_empty_string(self) -> None:
        assert isinstance(PROMPT_VIRTUAL_ASSISTANCE, str)
        assert len(PROMPT_VIRTUAL_ASSISTANCE.strip()) > 0

    # ------------------------------------------------------------------
    # Structural sections — every well-formed prompt needs these
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "section",
        [
            "# Identidade",
            "# Objetivo",
            "# Regras de Comportamento",
            "# Formato de Resposta",
            "# Restrições",
            "# Escalação para Humano",
        ],
    )
    def test_contains_required_section(self, section: str) -> None:
        assert section in PROMPT_VIRTUAL_ASSISTANCE, (
            f"Missing required section: {section}"
        )

    # ------------------------------------------------------------------
    # Identity — must mention company name
    # ------------------------------------------------------------------

    def test_mentions_company_name(self) -> None:
        assert "NovaMind" in PROMPT_VIRTUAL_ASSISTANCE

    # ------------------------------------------------------------------
    # Negative prompting — the NUNCA block
    # ------------------------------------------------------------------

    def test_contains_negative_prompting(self) -> None:
        assert "NUNCA" in PROMPT_VIRTUAL_ASSISTANCE

    @pytest.mark.parametrize(
        "forbidden_topic",
        [
            "preços",
            "dados internos",
            "promessas",
        ],
    )
    def test_negative_prompting_covers_topic(self, forbidden_topic: str) -> None:
        assert forbidden_topic in PROMPT_VIRTUAL_ASSISTANCE.lower(), (
            f"Negative prompting should mention: {forbidden_topic}"
        )

    # ------------------------------------------------------------------
    # Escalation triggers
    # ------------------------------------------------------------------

    def test_contains_escalation_triggers(self) -> None:
        prompt_lower = PROMPT_VIRTUAL_ASSISTANCE.lower()
        assert "escale" in prompt_lower or "escalação" in prompt_lower

    def test_escalation_mentions_human_handoff(self) -> None:
        assert "especialista" in PROMPT_VIRTUAL_ASSISTANCE.lower()

    # ------------------------------------------------------------------
    # Output format — structured response pattern
    # ------------------------------------------------------------------

    def test_defines_response_structure(self) -> None:
        assert "1." in PROMPT_VIRTUAL_ASSISTANCE
        assert "2." in PROMPT_VIRTUAL_ASSISTANCE
        assert "3." in PROMPT_VIRTUAL_ASSISTANCE
