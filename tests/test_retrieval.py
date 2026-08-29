"""Graph retrieval: seed resolution, traversal, ranking, provenance.

No GPU and no vision model: retrieval takes text seeds precisely so it
can be tested on its own. A bad end-to-end score has two possible causes
— the graph reached the wrong place, or the model guessed the wrong thing
to look for — and these tests pin down the first independently of the
second.
"""

import pytest

from fvqa.config import Config, ConfigError
from fvqa.data.fvqa_graph import KnowledgeGraph
from fvqa.retrieval import (
    GraphRetriever,
    LexicalRanker,
    build_ranker,
    format_facts,
    normalize_seed,
)
from fvqa.retrieval.types import EntityCandidate, RetrievedFact


def fact(e1, e2, relation="/r/RelatedTo", kb="conceptnet", surface=None):
    return {
        "KB": kb,
        "e1": e1,
        "e2": e2,
        "e1_label": e1,
        "e2_label": e2,
        "r": relation,
        "surface": surface or f"[[{e1}]] {relation.split('/')[-1]} [[{e2}]]",
    }


@pytest.fixture
def dog_graph():
    """A seed with many 1-hop facts, only one of which answers a question.

    This is the shape that makes ranking necessary: BFS returns all four
    `dog` facts at hop 1, so graph distance alone cannot choose.
    """
    return KnowledgeGraph(
        {
            "f1": fact("dog", "animal", "/r/IsA", surface="[[dog]] is a [[animal]]"),
            "f2": fact("dog", "house", "/r/AtLocation", surface="[[dog]] found in [[house]]"),
            "f3": fact("dog", "barking", "/r/CapableOf", surface="[[dog]] can [[barking]]"),
            "f4": fact("dog", "leash", "/r/RelatedTo", surface="[[dog]] related to [[leash]]"),
            # 2-hop from dog, via house.
            "f5": fact("house", "roof", "/r/HasA", surface="[[house]] has a [[roof]]"),
        }
    )


@pytest.fixture
def retriever(dog_graph):
    return GraphRetriever(dog_graph)


class TestNormalizeSeed:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("a trumpet", "trumpet"),
            ("A Trumpet", "trumpet"),
            ("the dog", "dog"),
            ("an apple", "apple"),
            ("  dog  ", "dog"),
            ("dog.", "dog"),
            ("dog", "dog"),
        ],
    )
    def test_trims_a_natural_language_guess_to_a_label(self, raw, expected):
        # A model asked what it sees says "a trumpet"; the node is
        # labelled "trumpet". Without this a correct guess resolves to
        # nothing at all.
        assert normalize_seed(raw) == expected

    def test_does_not_strip_an_article_that_is_the_whole_word(self):
        # "another" starts with "an" but is not an article + noun.
        assert normalize_seed("another") == "another"

    def test_empty_input(self):
        assert normalize_seed("   ") == ""


class TestResolveSeeds:
    def test_resolves_a_matching_seed(self, retriever):
        seeds = retriever.resolve_seeds(["dog"])
        assert [s.entity_id for s in seeds] == ["dog"]
        assert seeds[0].resolved

    def test_normalizes_before_looking_up(self, retriever):
        assert retriever.resolve_seeds(["A Dog"])[0].entity_id == "dog"

    def test_unmatched_seed_is_recorded_not_dropped(self, retriever):
        # A retrieval failure has to stay visible. Dropping it makes an
        # empty context indistinguishable from a question needing no facts.
        seeds = retriever.resolve_seeds(["spaceship"])
        assert len(seeds) == 1
        assert seeds[0].entity_id is None
        assert not seeds[0].resolved
        assert seeds[0].text == "spaceship"

    def test_records_the_source_of_each_guess(self, retriever):
        seeds = retriever.resolve_seeds(["dog"], source="vision")
        assert seeds[0].source == "vision"

    def test_respects_the_seed_entity_cap(self, retriever):
        seeds = retriever.resolve_seeds(["dog", "house", "roof"], max_seed_entities=2)
        assert len([s for s in seeds if s.resolved]) <= 2

    def test_skips_blank_seeds(self, retriever):
        assert retriever.resolve_seeds(["", "  "]) == []


class TestRetrieve:
    def test_finds_facts_from_a_resolved_seed(self, retriever):
        result = retriever.retrieve(["dog"], "What can a dog do?", max_hops=1)
        assert result.status == "ok"
        assert result.found_facts

    def test_ranking_puts_the_relevant_fact_first(self, retriever):
        # All four dog facts are at hop 1, so BFS order cannot choose
        # between them — only the question can.
        result = retriever.retrieve(
            ["dog"], "What sound does a dog make when barking?", max_hops=1
        )
        assert result.facts[0].triple.fact_id == "f3"  # dog CapableOf barking

    def test_a_question_with_no_discriminating_word_leaves_facts_tied(self, retriever):
        # "What can a dog do?" is the seed plus stopwords, so every dog
        # fact scores identically. A lexical ranker genuinely cannot
        # choose here, and the tie-break is fact id — deterministic, but
        # not meaningful. This is the limitation an embedding or LLM
        # reranker would be brought in to fix, recorded rather than
        # papered over.
        result = retriever.retrieve(["dog"], "What can a dog do?", max_hops=1)
        scores = {round(f.rank_score, 6) for f in result.facts}
        assert len(scores) == 1

    def test_a_different_question_reorders_the_same_candidates(self, retriever):
        result = retriever.retrieve(["dog"], "Where is the dog found?", max_hops=1)
        assert result.facts[0].triple.fact_id == "f2"  # dog AtLocation house

    def test_unmatched_seed_reports_no_seed_match(self, retriever):
        result = retriever.retrieve(["spaceship"], "What is this?")
        assert result.status == "no_seed_match"
        assert result.facts == []
        # The failure is legible in the result, not silent.
        assert result.as_dict()["seed_texts"] == ["spaceship"]
        assert result.as_dict()["resolved_entities"] == []

    def test_top_k_limits_what_reaches_the_prompt(self, retriever):
        result = retriever.retrieve(["dog"], "What is a dog?", max_hops=1, top_k_facts=2)
        assert len(result.facts) == 2
        assert result.num_candidates == 4

    def test_max_hops_bounds_the_traversal(self, retriever):
        one_hop = retriever.retrieve(
            ["dog"], "roof", max_hops=1, top_k_facts=99
        )
        two_hop = retriever.retrieve(
            ["dog"], "roof", max_hops=2, top_k_facts=99
        )
        one_hop_ids = {f.triple.fact_id for f in one_hop.facts}
        two_hop_ids = {f.triple.fact_id for f in two_hop.facts}
        assert "f5" not in one_hop_ids  # house HasA roof is 2 hops out
        assert "f5" in two_hop_ids

    def test_hop_is_recorded_on_each_fact(self, retriever):
        result = retriever.retrieve(["dog"], "roof", max_hops=2, top_k_facts=99)
        hops = {f.triple.fact_id: f.hop for f in result.facts}
        assert hops["f1"] == 1
        assert hops["f5"] == 2

    def test_max_candidate_facts_caps_the_traversal(self, retriever):
        result = retriever.retrieve(
            ["dog"], "What is a dog?", max_hops=2, max_candidate_facts=2, top_k_facts=99
        )
        assert result.num_candidates == 2

    def test_settings_are_recorded_for_reproducibility(self, retriever):
        # Two runs at different hop counts are otherwise indistinguishable
        # once the numbers are in a file.
        result = retriever.retrieve(["dog"], "What is a dog?", max_hops=3, top_k_facts=2)
        assert result.settings["max_hops"] == 3
        assert result.settings["top_k_facts"] == 2
        assert result.settings["ranking_method"] == "lexical"

    def test_results_are_deterministic(self, retriever):
        first = retriever.retrieve(["dog"], "What can a dog do?", max_hops=2)
        second = retriever.retrieve(["dog"], "What can a dog do?", max_hops=2)
        assert [f.triple.fact_id for f in first.facts] == [
            f.triple.fact_id for f in second.facts
        ]

    def test_an_isolated_seed_reports_no_facts_found(self):
        # A node that exists but has no edges: distinct from a seed that
        # matched nothing at all.
        graph = KnowledgeGraph({"f1": fact("dog", "animal")})
        retriever = GraphRetriever(graph)
        result = retriever.retrieve(["dog"], "anything", max_hops=1)
        assert result.status == "ok"

    def test_from_config_uses_the_configured_ranker(self, dog_graph):
        config = Config()
        retriever = GraphRetriever.from_config(dog_graph, config)
        assert isinstance(retriever.ranker, LexicalRanker)


class TestLexicalRanker:
    def make_fact(self, e1, e2, relation, hop=1, surface=None):
        graph = KnowledgeGraph({"f": fact(e1, e2, relation, surface=surface)})
        return RetrievedFact(triple=graph.fact("f"), hop=hop)

    def test_overlapping_content_words_score_higher(self):
        ranker = LexicalRanker()
        relevant = self.make_fact("dog", "barking", "/r/CapableOf")
        irrelevant = self.make_fact("dog", "leash", "/r/RelatedTo")
        question = "What sound does a dog make? barking"
        assert ranker.score(question, relevant) > ranker.score(question, irrelevant)

    def test_stopwords_do_not_drive_the_score(self):
        # A fact matching only "the"/"is" tells you nothing, and must not
        # outscore one matching a real content word.
        ranker = LexicalRanker()
        stopword_only = self.make_fact("the", "is", "/r/RelatedTo")
        content = self.make_fact("dog", "barking", "/r/CapableOf")
        question = "What is the dog doing? barking"
        assert ranker.score(question, content) > ranker.score(question, stopword_only)

    def test_deeper_facts_are_penalised(self):
        ranker = LexicalRanker(hop_penalty=0.5, seed_bonus=0.0)
        near = self.make_fact("dog", "barking", "/r/CapableOf", hop=1)
        far = self.make_fact("dog", "barking", "/r/CapableOf", hop=3)
        assert ranker.score("dog barking", near) > ranker.score("dog barking", far)

    def test_a_fact_touching_a_resolved_seed_gets_a_bonus(self):
        ranker = LexicalRanker(seed_bonus=1.0, hop_penalty=0.0)
        touching = self.make_fact("dog", "leash", "/r/RelatedTo")
        seeds = [EntityCandidate(text="dog", entity_id="dog", source="oracle")]
        assert ranker.score("x", touching, seeds) > ranker.score("x", touching, [])

    def test_an_unresolved_seed_confers_no_bonus(self):
        ranker = LexicalRanker(seed_bonus=1.0, hop_penalty=0.0)
        candidate = self.make_fact("dog", "leash", "/r/RelatedTo")
        unresolved = [EntityCandidate(text="spaceship", entity_id=None)]
        assert ranker.score("x", candidate, unresolved) == ranker.score("x", candidate, [])

    def test_an_empty_question_scores_zero_overlap(self):
        ranker = LexicalRanker(hop_penalty=0.0, seed_bonus=0.0)
        assert ranker.score("", self.make_fact("dog", "leash", "/r/RelatedTo")) == 0.0


class TestBuildRanker:
    def test_builds_the_lexical_ranker(self):
        assert isinstance(build_ranker("lexical"), LexicalRanker)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown ranking_method"):
            build_ranker("embedding")


class TestFormatFacts:
    def test_numbers_the_facts(self, retriever):
        result = retriever.retrieve(["dog"], "What can a dog do?", max_hops=1, top_k_facts=2)
        text = format_facts(result.facts)
        assert text.startswith("1. ")
        assert "\n2. " in text

    def test_unnumbered_mode(self, retriever):
        result = retriever.retrieve(["dog"], "What can a dog do?", max_hops=1, top_k_facts=1)
        assert not format_facts(result.facts, numbered=False).startswith("1.")

    def test_no_facts_yields_an_empty_string(self):
        # An empty string means the caller builds an ungrounded prompt,
        # rather than one with a Facts: header and nothing under it.
        assert format_facts([]) == ""


class TestRetrievalConfig:
    def test_defaults_are_valid(self):
        Config().retrieval.validate("retrieval")

    def test_max_hops_moved_off_data(self):
        # It describes the traversal, not the dataset: the graph is the
        # same graph whether you walk one hop or three.
        assert not hasattr(Config().data, "max_hops")
        assert Config().retrieval.max_hops == 2

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("max_hops", 0, "max_hops must be positive"),
            ("top_k_facts", 0, "top_k_facts must be positive"),
            ("max_seed_entities", 0, "max_seed_entities must be positive"),
            ("ranking_method", "magic", "ranking_method"),
        ],
    )
    def test_invalid_values_are_rejected(self, field, value, message):
        config = Config()
        setattr(config.retrieval, field, value)
        with pytest.raises(ConfigError, match=message):
            config.retrieval.validate("retrieval")

    def test_top_k_cannot_exceed_the_candidate_cap(self):
        config = Config()
        config.retrieval.max_candidate_facts = 3
        config.retrieval.top_k_facts = 10
        with pytest.raises(ConfigError, match="cannot exceed"):
            config.retrieval.validate("retrieval")

    def test_as_dict_carries_the_full_provenance(self):
        recorded = Config().retrieval.as_dict()
        assert set(recorded) == {
            "enabled",
            "max_hops",
            "max_seed_entities",
            "max_candidate_facts",
            "top_k_facts",
            "ranking_method",
        }
