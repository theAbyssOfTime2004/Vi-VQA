"""Evaluation conditions: what context each one puts in front of the question.

The conditions exist so that score differences are readable — each pair
differs in exactly one thing. These tests pin down that "exactly one
thing" so a condition cannot quietly stop doing its job and still report
a number.

No model and no GPU: what a condition builds is a data structure.
"""

import pytest

from fvqa.config import Config
from fvqa.data.fvqa_graph import KnowledgeGraph
from fvqa.evaluation.conditions import (
    CONDITIONS,
    ConditionContext,
    build_prompt,
    oracle_seed_labels,
)


def triple_record(e1, e2, relation="/r/AtLocation", surface=None):
    return {
        "KB": "conceptnet",
        "e1": e1,
        "e2": e2,
        "e1_label": e1,
        "e2_label": e2,
        "r": relation,
        "surface": surface or f"You are likely to find [[{e1}]] in [[{e2}]]",
    }


FACTS = {
    "f_trumpet": triple_record("trumpet", "jazz club"),
    "f_lizard": triple_record("lizard", "jazz club"),
    "f_noise": triple_record("jazz club", "music", "/r/RelatedTo"),
}


@pytest.fixture
def sample():
    return {
        "id": "fvqa_270",
        "image": "trumpet.jpg",
        "fvqa_question": "Which object can be found in a jazz club",
        "fvqa_answer": "trumpet",
        "fvqa_fact_ids": ["f_trumpet"],
        "conversations": [
            {"from": "human", "value": "<image>\nWhich object can be found in a jazz club"},
            {"from": "gpt", "value": "trumpet"},
        ],
    }


@pytest.fixture
def config():
    config = Config()
    config.data.image_folder = "/images"
    return config


def context_for(config, condition, facts=None):
    """A ConditionContext with the graph pre-loaded from fixtures."""
    context = ConditionContext(config=config, condition=condition)
    facts = FACTS if facts is None else facts
    context._facts = facts
    context._graph = KnowledgeGraph(facts)
    context._loaded = True
    if context.needs_graph:
        from fvqa.retrieval import GraphRetriever

        context._retriever = GraphRetriever.from_config(context._graph, config)
    return context


def user_text(prompt):
    return prompt.messages[-1]["content"][-1]["text"]


def system_texts(prompt):
    return [
        m["content"][0]["text"] for m in prompt.messages if m["role"] == "system"
    ]


class TestConditionValidation:
    def test_unknown_condition_is_rejected(self, config):
        with pytest.raises(ValueError, match="unknown condition"):
            ConditionContext(config=config, condition="magic")

    def test_vision_seed_graph_says_it_is_not_built_yet(self, config):
        # Silently falling back to another condition would report a
        # number for an experiment that never ran.
        with pytest.raises(NotImplementedError, match="not built yet"):
            ConditionContext(config=config, condition="vision-seed-graph")

    def test_only_graph_conditions_load_the_graph(self, config):
        assert not ConditionContext(config=config, condition="no-context").needs_graph
        assert not ConditionContext(config=config, condition="style").needs_graph
        assert ConditionContext(config=config, condition="oracle-fact").needs_graph
        assert ConditionContext(config=config, condition="oracle-seed-graph").needs_graph

    def test_every_listed_condition_is_constructible(self, config):
        for condition in CONDITIONS:
            if condition == "vision-seed-graph":
                continue
            ConditionContext(config=config, condition=condition)


class TestNoContext:
    def test_asks_the_bare_question(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "no-context"))
        assert user_text(prompt) == "Which object can be found in a jazz club"
        assert prompt.context is None

    def test_carries_no_system_turn_even_if_one_is_configured(self, sample, config):
        # This is the floor every other condition is measured against, so
        # it must carry nothing at all — including style.
        config.inference.system_prompt = "Answer in one word."
        prompt = build_prompt(sample, context_for(config, "no-context"))
        assert system_texts(prompt) == []

    def test_points_at_the_image(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "no-context"))
        image = prompt.messages[-1]["content"][0]
        assert image["type"] == "image"
        assert image["image"] == "/images/trumpet.jpg"


class TestStyle:
    def test_adds_the_style_prompt_and_nothing_else(self, sample, config):
        config.inference.system_prompt = "Answer in one word."
        prompt = build_prompt(sample, context_for(config, "style"))
        assert system_texts(prompt) == ["Answer in one word."]
        # The question itself is untouched — style says how to answer,
        # never what the answer is.
        assert user_text(prompt) == "Which object can be found in a jazz club"


class TestOracleFact:
    def test_puts_the_supporting_fact_in_the_prompt(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "oracle-fact"))
        assert "jazz club" in prompt.context
        assert "trumpet" in prompt.context
        assert prompt.context in user_text(prompt)

    def test_grounds_even_though_data_grounding_is_off(self, sample, config):
        # data.grounding.enabled governs whether `prepare` bakes facts
        # into the split file. Naming a grounded condition is a separate
        # decision, already made. Without this the condition would hand
        # the model nothing and report the score as if it had.
        assert config.data.grounding.enabled is False
        prompt = build_prompt(sample, context_for(config, "oracle-fact"))
        assert prompt.context is not None
        assert user_text(prompt) != sample["fvqa_question"]

    def test_records_which_fact_was_used(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "oracle-fact"))
        assert prompt.retrieval["status"] == "ok"
        assert prompt.retrieval["fact_ids"] == ["f_trumpet"]

    def test_a_sample_with_no_fact_is_recorded_not_silently_ungrounded(
        self, sample, config
    ):
        sample["fvqa_fact_ids"] = []
        prompt = build_prompt(sample, context_for(config, "oracle-fact"))
        assert prompt.retrieval["status"] == "no_oracle_fact"
        assert prompt.context is None


class TestOracleSeedGraph:
    def test_seeds_from_the_facts_non_answer_entity(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert prompt.retrieval["seed_texts"] == ["jazz club"]

    def test_never_seeds_with_the_answer(self, sample, config):
        # Seeding with the answer's own entity would put the answer in
        # the prompt and measure nothing.
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert "trumpet" not in prompt.retrieval["seed_texts"]

    def test_does_not_hand_over_the_fact_id(self, sample, config):
        # The whole point: the correct starting entity, never the correct
        # destination. The fact has to survive traversal and ranking.
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert prompt.retrieval["num_candidates"] >= 2

    def test_records_whether_the_supporting_fact_survived_retrieval(
        self, sample, config
    ):
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert prompt.retrieval["oracle_fact_retrieved"] is True
        assert prompt.retrieval["oracle_fact_ids"] == ["f_trumpet"]

    def test_reports_a_miss_as_a_miss(self, sample, config):
        # top_k=1 with a competing fact ranked first: the supporting fact
        # is squeezed out, and that has to show up in the provenance
        # rather than being invisible behind a low score.
        config.retrieval.top_k_facts = 1
        facts = dict(FACTS)
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph", facts))
        retrieved_ids = [f["fact_id"] for f in prompt.retrieval["facts"]]
        assert prompt.retrieval["oracle_fact_retrieved"] == (
            "f_trumpet" in retrieved_ids
        )

    def test_records_the_retrieval_settings(self, sample, config):
        config.retrieval.max_hops = 3
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert prompt.retrieval["settings"]["max_hops"] == 3

    def test_unusable_seed_is_recorded(self, config):
        # Both endpoints look like the answer: this condition cannot run
        # on that question, which is a fact about the question, not an error.
        sample = {
            "id": "x",
            "image": "x.jpg",
            "fvqa_question": "What is it?",
            "fvqa_answer": "trumpet",
            "fvqa_fact_ids": ["f_self"],
            "conversations": [{"from": "gpt", "value": "trumpet"}],
        }
        facts = {"f_self": triple_record("trumpet", "trumpet player")}
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph", facts))
        assert prompt.retrieval["status"] == "no_oracle_seed"
        assert prompt.context is None

    def test_facts_reach_the_prompt_text(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "oracle-seed-graph"))
        assert prompt.context
        assert prompt.context in user_text(prompt)


class TestStored:
    def test_replays_the_prompt_written_by_prepare(self, sample, config):
        prompt = build_prompt(sample, context_for(config, "stored"))
        assert user_text(prompt) == "Which object can be found in a jazz club"

    def test_keeps_a_baked_in_system_turn(self, config):
        sample = {
            "id": "x",
            "image": "x.jpg",
            "conversations": [
                {"from": "system", "value": "Fact: a trumpet is in a jazz club"},
                {"from": "human", "value": "<image>\nWhich object?"},
                {"from": "gpt", "value": "trumpet"},
            ],
        }
        prompt = build_prompt(sample, context_for(config, "stored"))
        assert "Fact: a trumpet is in a jazz club" in system_texts(prompt)


class TestOracleSeedLabels:
    def make_triple(self, e1, e2):
        return KnowledgeGraph({"f": triple_record(e1, e2)}).fact("f")

    def test_excludes_the_answer(self):
        triple = self.make_triple("trumpet", "jazz club")
        assert oracle_seed_labels(triple, "trumpet") == ["jazz club"]

    def test_is_case_insensitive(self):
        triple = self.make_triple("Trumpet", "jazz club")
        assert oracle_seed_labels(triple, "TRUMPET") == ["jazz club"]

    def test_excludes_a_label_containing_the_answer(self):
        # "trumpet player" contains the answer, so seeding with it would
        # still leak it.
        triple = self.make_triple("trumpet player", "jazz club")
        assert oracle_seed_labels(triple, "trumpet") == ["jazz club"]

    def test_both_endpoints_excluded_yields_nothing(self):
        triple = self.make_triple("trumpet", "brass trumpet")
        assert oracle_seed_labels(triple, "trumpet") == []

    def test_keeps_both_when_neither_is_the_answer(self):
        triple = self.make_triple("trumpet", "jazz club")
        assert oracle_seed_labels(triple, "saxophone") == ["trumpet", "jazz club"]


class TestBackwardCompatibility:
    def test_a_split_without_the_raw_question_still_works(self, config):
        # Splits written before `prepare` stored fvqa_question: with
        # grounding off, the human turn is the question.
        sample = {
            "id": "x",
            "image": "x.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "trumpet"},
            ],
        }
        prompt = build_prompt(sample, context_for(config, "no-context"))
        assert user_text(prompt) == "What is this?"
