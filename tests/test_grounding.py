"""Knowledge grounding from the dataset's description field."""

from fvqa.config import GroundingConfig
from fvqa.data.grounding import apply_grounding, truncate_description

DESCRIPTION = (
    "Chợ Bến Thành nằm ở quận 1. Chợ được xây năm 1912. "
    "Đây là biểu tượng của thành phố Hồ Chí Minh."
)


class TestTruncateDescription:
    def test_short_text_is_untouched(self):
        assert truncate_description("Ngắn.", 100) == "Ngắn."

    def test_cuts_on_a_sentence_boundary(self):
        result = truncate_description(DESCRIPTION, 60)
        # A dangling clause invites the model to complete it; whole
        # sentences do not.
        assert result.endswith(".")
        assert len(result) <= 60

    def test_falls_back_to_a_hard_cut(self):
        # No sentence fits, but a truncated fact still beats no context.
        long_sentence = "x" * 200
        assert len(truncate_description(long_sentence, 50)) == 50

    def test_surrounding_whitespace_is_stripped(self):
        assert truncate_description("  Ngắn.  ", 100) == "Ngắn."


class TestApplyGrounding:
    def test_disabled_returns_the_bare_question(self):
        prompt = apply_grounding("Đây là gì?", DESCRIPTION, GroundingConfig(enabled=False))
        assert prompt.question == "Đây là gì?"
        assert prompt.system is None

    def test_missing_description_returns_the_bare_question(self):
        config = GroundingConfig(enabled=True)
        for empty in (None, "", "   "):
            prompt = apply_grounding("Đây là gì?", empty, config)
            assert prompt.question == "Đây là gì?"
            assert prompt.system is None

    def test_prefix_mode_combines_context_and_question(self):
        prompt = apply_grounding("Đây là gì?", DESCRIPTION, GroundingConfig(enabled=True))
        assert "Chợ Bến Thành" in prompt.question
        assert "Đây là gì?" in prompt.question
        assert prompt.system is None

    def test_system_mode_keeps_the_question_in_the_user_turn(self):
        config = GroundingConfig(enabled=True, mode="system")
        prompt = apply_grounding("Đây là gì?", DESCRIPTION, config)
        assert prompt.question == "Đây là gì?"
        assert "Chợ Bến Thành" in prompt.system
        assert "Đây là gì?" not in prompt.system

    def test_context_respects_max_chars(self):
        config = GroundingConfig(enabled=True, max_chars=40)
        prompt = apply_grounding("Đây là gì?", DESCRIPTION, config)
        assert "biểu tượng" not in prompt.question

    def test_custom_template_is_used(self):
        config = GroundingConfig(enabled=True, template="CTX[{description}] Q[{question}]")
        prompt = apply_grounding("Đây là gì?", "Chợ.", config)
        assert prompt.question == "CTX[Chợ.] Q[Đây là gì?]"
