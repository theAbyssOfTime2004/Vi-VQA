"""Generative VQA metrics.

Exact match alone is close to useless here: answers average ~49 characters
of free-form Vietnamese, so a correct answer phrased differently scores
zero. These metrics grade partial credit instead.

Everything is implemented in the standard library. `nltk` and
`rouge-score` would each pull a download step or a heavyweight dependency
into a container that already has to fit an 8B model.

Scales, so numbers are never compared across incompatible ranges:
    exact_match, similarity, bleu, rouge_l -> 0-100
    cider                                  -> 0-10 (its native scale)
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable, Iterable, Sequence

__all__ = [
    "AVAILABLE_METRICS",
    "cider",
    "compute_metrics",
    "corpus_bleu",
    "exact_match",
    "normalize_text",
    "rouge_l",
    "similarity",
    "tokenize",
]

AVAILABLE_METRICS = ("exact_match", "similarity", "bleu", "rouge_l", "cider")

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize a Vietnamese answer for comparison.

    NFC normalization is not cosmetic: "à" can be stored either as one
    codepoint or as "a" plus a combining grave accent. The two render
    identically, compare unequal, and the dataset contains both — without
    this step exact match silently under-reports.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokens of the normalized text.

    Vietnamese is written with spaces between *syllables*, not words, so
    these are syllable tokens. That is what every ViTextVQA-style
    benchmark scores on, and it keeps the metric tokenizer-free.
    """
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# Per-sample metrics
# --------------------------------------------------------------------------


def exact_match(prediction: str, reference: str) -> float:
    """1.0 when the normalized strings are identical, else 0.0."""
    return float(normalize_text(prediction) == normalize_text(reference))


def similarity(prediction: str, reference: str) -> float:
    """Character-level similarity ratio in [0, 1]."""
    return SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio()


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure in [0, 1].

    beta=1.2 weights recall slightly above precision, the convention used
    by the ROUGE package and by captioning benchmarks.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta**2
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


# --------------------------------------------------------------------------
# Corpus metrics
# --------------------------------------------------------------------------


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    max_n: int = 4,
) -> float:
    """Corpus-level BLEU-4 in [0, 1], one reference per prediction.

    Corpus-level rather than an average of per-sentence scores: short VQA
    answers frequently contain no 4-gram at all, which makes sentence BLEU
    collapse to zero and the average meaningless.

    Higher orders with no match are smoothed rather than zeroing the
    score, so a low BLEU still ranks two models apart. Predictions sharing
    no unigram at all with the references score exactly 0.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have equal length, "
            f"got {len(predictions)} and {len(references)}"
        )
    if not predictions:
        return 0.0

    matches = [0] * max_n
    totals = [0] * max_n
    pred_length = 0
    ref_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)
            totals[n - 1] += max(len(pred_tokens) - n + 1, 0)
            for ngram, count in pred_ngrams.items():
                matches[n - 1] += min(count, ref_ngrams.get(ngram, 0))

    # No unigram in common means the predictions are unrelated to the
    # references, and BLEU should say 0 rather than the small floor that
    # smoothing would otherwise leave behind.
    if matches[0] == 0:
        return 0.0

    # Geometric mean of the modified precisions. A zero at a higher order
    # would zero the whole corpus score even when the lower orders show
    # real overlap, so floor it (NLTK "method 1" smoothing).
    log_precision_sum = 0.0
    for n in range(max_n):
        if totals[n] == 0:
            return 0.0
        precision = matches[n] / totals[n]
        if precision == 0.0:
            precision = 1.0 / (2 * totals[n])
        log_precision_sum += math.log(precision)
    geometric_mean = math.exp(log_precision_sum / max_n)

    if pred_length == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_length > ref_length else math.exp(1 - ref_length / pred_length)
    return brevity_penalty * geometric_mean


def cider(
    predictions: Sequence[str],
    references: Sequence[str],
    max_n: int = 4,
    sigma: float = 6.0,
) -> float:
    """CIDEr-D in [0, 10], one reference per prediction.

    Document frequencies come from the reference set being evaluated, as
    in the original coco-caption implementation. The consequence worth
    knowing: scores are only comparable between runs evaluated on the same
    set of samples, so keep `evaluation.num_samples` fixed when comparing.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have equal length, "
            f"got {len(predictions)} and {len(references)}"
        )
    if not predictions:
        return 0.0

    pred_tokens = [tokenize(p) for p in predictions]
    ref_tokens = [tokenize(r) for r in references]

    num_docs = len(references)
    document_frequency: Counter[tuple[str, ...]] = Counter()
    for tokens in ref_tokens:
        seen: set[tuple[str, ...]] = set()
        for n in range(1, max_n + 1):
            seen.update(_ngrams(tokens, n))
        document_frequency.update(seen)

    log_num_docs = math.log(max(num_docs, 1))

    def tf_idf(tokens: Sequence[str], n: int) -> tuple[dict[tuple[str, ...], float], float]:
        counts = _ngrams(tokens, n)
        vector: dict[tuple[str, ...], float] = {}
        norm = 0.0
        for ngram, count in counts.items():
            idf = log_num_docs - math.log(max(document_frequency.get(ngram, 0), 1))
            value = count * idf
            vector[ngram] = value
            norm += value**2
        return vector, math.sqrt(norm)

    scores = []
    for pred, ref in zip(pred_tokens, ref_tokens):
        # Length penalty discourages padding the answer with filler.
        length_delta = len(pred) - len(ref)
        penalty = math.exp(-(length_delta**2) / (2 * sigma**2))

        per_order = []
        for n in range(1, max_n + 1):
            pred_vector, pred_norm = tf_idf(pred, n)
            ref_vector, ref_norm = tf_idf(ref, n)
            if pred_norm == 0.0 or ref_norm == 0.0:
                per_order.append(0.0)
                continue
            overlap = sum(
                # min() clips repeated n-grams, the "-D" in CIDEr-D: it
                # stops a candidate gaming the score by repeating a
                # high-idf phrase.
                min(value, ref_vector.get(ngram, 0.0)) * ref_vector.get(ngram, 0.0)
                for ngram, value in pred_vector.items()
            )
            per_order.append(overlap / (pred_norm * ref_norm) * penalty)
        scores.append(10.0 * sum(per_order) / max_n)

    return sum(scores) / len(scores)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

_PER_SAMPLE: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "similarity": similarity,
    "rouge_l": rouge_l,
}


def compute_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics: Iterable[str] = AVAILABLE_METRICS,
) -> dict[str, float]:
    """Score predictions against references.

    Raises:
        ValueError: unknown metric name, or mismatched input lengths.
    """
    metrics = list(metrics)
    unknown = [m for m in metrics if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {unknown}. Available: {list(AVAILABLE_METRICS)}")
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have equal length, "
            f"got {len(predictions)} and {len(references)}"
        )

    results: dict[str, float] = {}
    for name in metrics:
        if name in _PER_SAMPLE:
            fn = _PER_SAMPLE[name]
            if predictions:
                total = sum(fn(p, r) for p, r in zip(predictions, references))
                results[name] = 100.0 * total / len(predictions)
            else:
                results[name] = 0.0
        elif name == "bleu":
            results[name] = 100.0 * corpus_bleu(predictions, references)
        elif name == "cider":
            results[name] = cider(predictions, references)
    return results
