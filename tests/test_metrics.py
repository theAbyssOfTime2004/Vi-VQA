"""Evaluation metrics."""

import unicodedata

import pytest

from vivqa.evaluation.metrics import (
    AVAILABLE_METRICS,
    cider,
    compute_metrics,
    corpus_bleu,
    exact_match,
    normalize_text,
    rouge_l,
    similarity,
    tokenize,
)

REFERENCES = [
    "Đây là khu chợ.",
    "Đây là siêu thị VinMart.",
    "Bức ảnh này chụp ở chợ Bến Thành.",
]


class TestNormalization:
    def test_nfc_and_nfd_compare_equal(self):
        # "à" can be one codepoint or "a" plus a combining accent. Both
        # occur in the dataset and render identically; without NFC they
        # compare unequal and exact match silently under-reports.
        text = "Đây là khu chợ"
        assert normalize_text(unicodedata.normalize("NFD", text)) == normalize_text(text)

    def test_case_and_punctuation_are_ignored(self):
        assert normalize_text("Đây LÀ khu chợ!!!") == normalize_text("đây là khu chợ")

    def test_whitespace_is_collapsed(self):
        assert normalize_text("  đây   là  chợ ") == "đây là chợ"

    def test_diacritics_are_preserved(self):
        # Stripping tone marks would collapse distinct Vietnamese words.
        assert normalize_text("chợ") != normalize_text("cho")

    def test_tokenize_empty_string(self):
        assert tokenize("") == []
        assert tokenize("   !!!  ") == []


class TestPerSampleMetrics:
    def test_exact_match_ignores_case_and_punctuation(self):
        assert exact_match("Đây là khu chợ.", "đây là khu chợ") == 1.0

    def test_exact_match_rejects_different_answers(self):
        assert exact_match("Đây là siêu thị.", "Đây là khu chợ.") == 0.0

    def test_similarity_is_bounded(self):
        assert similarity("Đây là khu chợ.", "Đây là khu chợ.") == 1.0
        assert 0.0 <= similarity("hoàn toàn khác", "Đây là khu chợ.") < 1.0

    def test_rouge_l_rewards_partial_overlap(self):
        # Exact match scores this zero; the answer is most of the way there.
        partial = rouge_l("Đây là khu chợ lớn", "Đây là khu chợ")
        assert 0.0 < partial < 1.0
        assert rouge_l("Đây là khu chợ", "Đây là khu chợ") == 1.0

    def test_rouge_l_handles_empty_input(self):
        assert rouge_l("", "Đây là khu chợ") == 0.0
        assert rouge_l("Đây là khu chợ", "") == 0.0

    def test_rouge_l_respects_word_order(self):
        in_order = rouge_l("a b c d", "a b c d")
        shuffled = rouge_l("d c b a", "a b c d")
        assert shuffled < in_order


class TestCorpusMetrics:
    def test_bleu_is_one_for_identical_corpora(self):
        assert corpus_bleu(REFERENCES, REFERENCES) == pytest.approx(1.0)

    def test_bleu_penalises_short_answers(self):
        truncated = ["Đây là", "Đây là siêu", "Bức ảnh này"]
        assert corpus_bleu(truncated, REFERENCES) < corpus_bleu(REFERENCES, REFERENCES)

    def test_bleu_is_zero_without_any_overlap(self):
        unrelated = ["xyz abc def ghi"] * 3
        assert corpus_bleu(unrelated, REFERENCES) == pytest.approx(0.0, abs=1e-6)

    def test_bleu_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            corpus_bleu(["a"], ["a", "b"])

    def test_cider_peaks_at_ten_for_identical_corpora(self):
        assert cider(REFERENCES, REFERENCES) == pytest.approx(10.0)

    def test_cider_drops_for_wrong_answers(self):
        wrong = list(reversed(REFERENCES))
        assert cider(wrong, REFERENCES) < cider(REFERENCES, REFERENCES)

    def test_cider_does_not_reward_padding(self):
        # The length penalty is what stops a candidate inflating its score
        # by appending high-idf filler.
        padded = [ref + " " + ref for ref in REFERENCES]
        assert cider(padded, REFERENCES) < cider(REFERENCES, REFERENCES)

    def test_empty_corpus_scores_zero(self):
        assert corpus_bleu([], []) == 0.0
        assert cider([], []) == 0.0


class TestComputeMetrics:
    def test_perfect_predictions_max_every_metric(self):
        scores = compute_metrics(REFERENCES, REFERENCES)
        assert scores["exact_match"] == pytest.approx(100.0)
        assert scores["similarity"] == pytest.approx(100.0)
        assert scores["bleu"] == pytest.approx(100.0)
        assert scores["rouge_l"] == pytest.approx(100.0)
        assert scores["cider"] == pytest.approx(10.0)

    def test_partial_credit_where_exact_match_gives_none(self):
        # The whole point of these metrics: a right answer phrased
        # differently should not score zero.
        close = ["Đây là cái chợ", "Siêu thị VinMart", "Ảnh chụp tại chợ Bến Thành"]
        scores = compute_metrics(close, REFERENCES)
        assert scores["exact_match"] == 0.0
        assert scores["rouge_l"] > 40.0
        assert scores["similarity"] > 50.0

    def test_metric_subset_is_honoured(self):
        assert set(compute_metrics(REFERENCES, REFERENCES, ["bleu"])) == {"bleu"}

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError, match="unknown metric"):
            compute_metrics(REFERENCES, REFERENCES, ["meteor"])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_metrics(["a"], ["a", "b"])

    def test_every_advertised_metric_is_computable(self):
        scores = compute_metrics(REFERENCES, REFERENCES, AVAILABLE_METRICS)
        assert set(scores) == set(AVAILABLE_METRICS)
