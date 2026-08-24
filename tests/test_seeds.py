"""Vision seeding: parsing what the model said, caching it, loosening it.

The vision model's reply is the least controllable input in the pipeline —
it is free text from a model that was asked nicely for JSON. These tests
cover the ways that goes wrong, because each one turns into a retrieval
failure that looks like the graph's fault.
"""

import json

import pytest

from fvqa.retrieval import (
    ManualSeedProvider,
    QwenVisionSeedProvider,
    SeedCache,
    normalize_seed,
    parse_seed_response,
    seed_variants,
    singularize,
)


class TestParseSeedResponse:
    def test_clean_json(self):
        assert parse_seed_response('{"entities": ["trumpet", "jazz club"]}') == [
            "trumpet",
            "jazz club",
        ]

    def test_json_in_a_code_fence(self):
        reply = '```json\n{"entities": ["trumpet"]}\n```'
        assert parse_seed_response(reply) == ["trumpet"]

    def test_json_after_a_preamble(self):
        reply = 'Sure! Here are the entities:\n{"entities": ["trumpet"]}'
        assert parse_seed_response(reply) == ["trumpet"]

    def test_deduplicates_case_insensitively_keeping_order(self):
        reply = '{"entities": ["Trumpet", "trumpet", "TRUMPET", "jazz club"]}'
        assert parse_seed_response(reply) == ["Trumpet", "jazz club"]

    def test_caps_at_five(self):
        entities = [f"thing{i}" for i in range(20)]
        reply = json.dumps({"entities": entities})
        assert len(parse_seed_response(reply)) == 5

    def test_ignores_non_string_entries(self):
        assert parse_seed_response('{"entities": ["trumpet", 42, null]}') == ["trumpet"]

    def test_falls_back_to_lines_when_there_is_no_json(self):
        # Losing the seed list entirely turns into a retrieval failure
        # that looks like the graph's fault, so a bad recovery beats none.
        assert parse_seed_response("- trumpet\n- jazz club") == ["trumpet", "jazz club"]

    def test_falls_back_on_malformed_json(self):
        assert parse_seed_response('{"entities": ["trumpet",}') == []

    def test_empty_and_none_are_safe(self):
        assert parse_seed_response("") == []
        assert parse_seed_response(None) == []

    def test_a_json_object_without_entities_yields_nothing(self):
        assert parse_seed_response('{"answer": "trumpet"}') == []


class TestSingularize:
    @pytest.mark.parametrize(
        "plural, singular",
        [
            ("carrots", "carrot"),
            ("trumpets", "trumpet"),
            ("berries", "berry"),
            ("benches", "bench"),
            ("dishes", "dish"),
            ("glasses", "glass"),
        ],
    )
    def test_strips_common_plurals(self, plural, singular):
        assert singularize(plural) == singular

    @pytest.mark.parametrize("word", ["grass", "bus", "tennis", "cat", "is"])
    def test_leaves_non_plurals_alone(self, word):
        # Getting one of these wrong costs a fallback attempt that finds
        # nothing — never a wrong answer — so the rules stay conservative.
        assert singularize(word) == word


class TestSeedVariants:
    def test_tight_form_comes_first(self):
        assert seed_variants("a Trumpet")[0] == "trumpet"

    def test_adds_the_singular(self):
        assert "trumpet" in seed_variants("trumpets")

    def test_adds_the_head_noun_of_a_phrase(self):
        # "orange vegetables" reaching `vegetable` beats reaching nothing.
        variants = seed_variants("orange vegetables")
        assert "orange vegetables" == variants[0]
        assert "vegetable" in variants

    def test_no_duplicate_variants(self):
        variants = seed_variants("trumpet")
        assert len(variants) == len(set(variants))

    def test_empty_seed_yields_nothing(self):
        assert seed_variants("   ") == []


class TestManualSeedProvider:
    def test_returns_its_fixed_list(self):
        provider = ManualSeedProvider(["trumpet"])
        assert provider.seeds("/any.jpg", "any question") == ["trumpet"]

    def test_returns_a_copy(self):
        provider = ManualSeedProvider(["trumpet"])
        provider.seeds("/a.jpg", "q").append("mutated")
        assert provider.seeds("/a.jpg", "q") == ["trumpet"]


class TestSeedCache:
    def test_round_trips(self, tmp_path):
        cache = SeedCache(str(tmp_path), "Qwen/Qwen3-VL-8B-Instruct")
        cache.put("img1.jpg", "q1", ["trumpet"], raw='{"entities": ["trumpet"]}')
        assert cache.get("img1.jpg", "q1") == ["trumpet"]

    def test_misses_return_none(self, tmp_path):
        cache = SeedCache(str(tmp_path), "m")
        assert cache.get("nope.jpg", "q") is None

    def test_keyed_by_model(self, tmp_path):
        # Different models see different things; sharing a cache between
        # them would attribute one model's guesses to another.
        SeedCache(str(tmp_path), "model-a").put("i.jpg", "q", ["trumpet"])
        assert SeedCache(str(tmp_path), "model-b").get("i.jpg", "q") is None

    def test_keeps_the_raw_reply_for_tracing(self, tmp_path):
        cache = SeedCache(str(tmp_path), "m")
        cache.put("i.jpg", "q", ["trumpet"], raw="model said this")
        stored = json.loads(open(cache.path_for("i.jpg", "q")).read())
        assert stored["raw"] == "model said this"

    def test_a_corrupt_entry_is_ignored_not_fatal(self, tmp_path):
        cache = SeedCache(str(tmp_path), "m")
        path = cache.path_for("i.jpg", "q")
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write("{not json")
        assert cache.get("i.jpg", "q") is None

    def test_ids_with_separators_cannot_escape_the_cache_root(self, tmp_path):
        # Image ids are filenames from the dataset and question ids are
        # built from them, so neither is attacker-controlled here — but a
        # cache that writes outside its own root on odd input is a bug
        # waiting for the first dataset with a slash in a filename.
        import os

        cache = SeedCache(str(tmp_path), "org/model")
        path = os.path.realpath(cache.path_for("../../etc/passwd", "../q"))
        root = os.path.realpath(str(tmp_path))
        assert os.path.commonpath([path, root]) == root


class RecordingModel:
    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.reply


class TestQwenVisionSeedProvider:
    def test_asks_the_model_and_parses_the_reply(self):
        model = RecordingModel('{"entities": ["trumpet"]}')
        provider = QwenVisionSeedProvider(model)
        assert provider.seeds("/i.jpg", "What is this?") == ["trumpet"]

    def test_generates_greedily(self):
        # Seeds must not change between runs of an otherwise identical
        # experiment.
        model = RecordingModel('{"entities": ["trumpet"]}')
        QwenVisionSeedProvider(model).seeds("/i.jpg", "q")
        assert model.calls[0][1]["temperature"] == 0.0

    def test_the_prompt_carries_the_image_and_the_question(self):
        model = RecordingModel('{"entities": []}')
        QwenVisionSeedProvider(model).seeds("/images/trumpet.jpg", "Which object?")
        content = model.calls[0][0][-1]["content"]
        assert content[0] == {"type": "image", "image": "/images/trumpet.jpg"}
        assert "Which object?" in content[-1]["text"]

    def test_the_prompt_asks_for_objects_scenes_and_actions(self):
        # FVQA questions are tagged obj/scn/act; asking only about objects
        # cannot seed the scene and action questions at all.
        model = RecordingModel('{"entities": []}')
        QwenVisionSeedProvider(model).seeds("/i.jpg", "q")
        text = model.calls[0][0][-1]["content"][-1]["text"].lower()
        assert "objects" in text and "scenes" in text and "actions" in text

    def test_a_model_error_yields_no_seeds_rather_than_ending_the_run(self):
        class Failing:
            def generate(self, messages, **kwargs):
                raise RuntimeError("CUDA OOM")

        assert QwenVisionSeedProvider(Failing()).seeds("/i.jpg", "q") == []

    def test_uses_the_cache_on_the_second_call(self, tmp_path):
        model = RecordingModel('{"entities": ["trumpet"]}')
        cache = SeedCache(str(tmp_path), "m")
        provider = QwenVisionSeedProvider(model, cache=cache)
        first = provider.seeds("/i.jpg", "q", image_id="i.jpg", question_id="q1")
        second = provider.seeds("/i.jpg", "q", image_id="i.jpg", question_id="q1")
        assert first == second == ["trumpet"]
        assert len(model.calls) == 1

    def test_caches_an_empty_result_too(self, tmp_path):
        # A model that named nothing will name nothing again; re-asking
        # costs a generate call to learn the same thing.
        model = RecordingModel('{"entities": []}')
        cache = SeedCache(str(tmp_path), "m")
        provider = QwenVisionSeedProvider(model, cache=cache)
        provider.seeds("/i.jpg", "q", image_id="i.jpg", question_id="q1")
        provider.seeds("/i.jpg", "q", image_id="i.jpg", question_id="q1")
        assert len(model.calls) == 1

    def test_without_ids_the_cache_is_bypassed(self, tmp_path):
        model = RecordingModel('{"entities": ["trumpet"]}')
        provider = QwenVisionSeedProvider(model, cache=SeedCache(str(tmp_path), "m"))
        provider.seeds("/i.jpg", "q")
        provider.seeds("/i.jpg", "q")
        assert len(model.calls) == 2


class TestSeedNormalizationEndToEnd:
    @pytest.mark.parametrize(
        "guess", ["a trumpet", "The Trumpet", "trumpet.", "  trumpet  "]
    )
    def test_common_vlm_phrasings_normalize_to_the_label(self, guess):
        assert normalize_seed(guess) == "trumpet"
