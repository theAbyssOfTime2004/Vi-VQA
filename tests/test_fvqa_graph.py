"""KnowledgeGraph: adjacency construction, BFS, shortest-path.

This is the one place in the project that does real graph traversal.
Everywhere else, grounding hands the model a piece of text chosen in
advance; here nothing is chosen in advance — the graph is built from raw
triples and walked with actual BFS.

Fixtures mirror FVQA's real field names (`KB`, `e1_label`, `e2_label`,
`surface`, `r`, `e1`, `e2`) verified by downloading and inspecting the
release directly, not by trusting its README (which understates the
triple count and calls `fact` a single ID when it is actually a list).
"""

import pytest

from fvqa.data.fvqa_graph import KnowledgeGraph, Triple


def fact(e1, e2, e1_label=None, e2_label=None, kb="conceptnet", surface=None):
    """One triple in FVQA's real shape, defaults filled from the IDs."""
    return {
        "KB": kb,
        "e1": e1,
        "e2": e2,
        "e1_label": e1_label or e1,
        "e2_label": e2_label or e2,
        "r": "/r/RelatedTo",
        "surface": surface or f"[[{e1_label or e1}]] related to [[{e2_label or e2}]]",
    }


@pytest.fixture
def chain_graph():
    """a - b - c - d - e, plus a shortcut branch b - f.

    Distances from a: b=1, {c,f}=2, d=3, e=4.
    """
    return KnowledgeGraph(
        {
            "f1": fact("a", "b"),
            "f2": fact("b", "c"),
            "f3": fact("c", "d"),
            "f4": fact("d", "e"),
            "f5": fact("b", "f"),
        }
    )


class TestConstruction:
    def test_counts(self, chain_graph):
        assert len(chain_graph) == 5
        assert chain_graph.num_entities == 6

    def test_empty_graph(self):
        g = KnowledgeGraph({})
        assert len(g) == 0
        assert g.num_entities == 0
        assert g.bfs(["anything"], max_hops=1) == []

    def test_unknown_entity_has_no_neighbors(self, chain_graph):
        assert chain_graph.neighbors("does-not-exist") == []

    def test_has_entity(self, chain_graph):
        assert chain_graph.has_entity("a")
        assert not chain_graph.has_entity("zzz")


class TestTripleOther:
    def test_returns_the_opposite_endpoint(self):
        t = Triple("f1", "a", "b", "a", "b", "/r/RelatedTo", "conceptnet", "...")
        assert t.other("a") == "b"
        assert t.other("b") == "a"

    def test_rejects_an_id_that_is_not_an_endpoint(self):
        t = Triple("f1", "a", "b", "a", "b", "/r/RelatedTo", "conceptnet", "...")
        with pytest.raises(ValueError, match="not an endpoint"):
            t.other("c")


class TestNeighbors:
    def test_returns_every_triple_touching_the_entity(self, chain_graph):
        fact_ids = {t.fact_id for t in chain_graph.neighbors("b")}
        # b sits on three edges: to a, to c, and the f5 shortcut to f.
        assert fact_ids == {"f1", "f2", "f5"}

    def test_undirected_lookup_from_either_endpoint(self, chain_graph):
        # e1='a', e2='b' in the fixture; neighbors('b') must still see f1
        # even though b is the e2 side, not the e1 side, of that triple.
        assert "f1" in {t.fact_id for t in chain_graph.neighbors("a")}
        assert "f1" in {t.fact_id for t in chain_graph.neighbors("b")}


class TestBFS:
    def test_one_hop_reaches_only_direct_neighbors(self, chain_graph):
        result = {t.fact_id for t in chain_graph.bfs(["a"], max_hops=1)}
        assert result == {"f1"}

    def test_two_hops_reaches_the_branch_and_the_next_link(self, chain_graph):
        result = {t.fact_id for t in chain_graph.bfs(["a"], max_hops=2)}
        assert result == {"f1", "f2", "f5"}

    def test_bfs_order_is_breadth_first_not_depth_first(self, chain_graph):
        # f2/f5 (2 hops) must both appear before f3 (3 hops) in the result,
        # regardless of which of f2/f5 comes first between themselves.
        order = [t.fact_id for t in chain_graph.bfs(["a"], max_hops=3)]
        assert order.index("f3") > order.index("f2")
        assert order.index("f3") > order.index("f5")

    def test_a_fact_beyond_max_hops_is_excluded(self, chain_graph):
        # e is 4 edges from a; max_hops=3 must not include f4 (the d-e edge).
        result = {t.fact_id for t in chain_graph.bfs(["a"], max_hops=3)}
        assert "f4" not in result
        result = {t.fact_id for t in chain_graph.bfs(["a"], max_hops=4)}
        assert "f4" in result

    def test_each_fact_appears_at_most_once(self, chain_graph):
        # b is reachable from a and (trivially) is itself a seed too; the
        # a-b edge must not be double-counted.
        result = [t.fact_id for t in chain_graph.bfs(["a", "b"], max_hops=2)]
        assert len(result) == len(set(result))

    def test_multiple_seeds_are_explored_together(self, chain_graph):
        # Seeding from d directly reaches both its edges (to c and to e) at
        # hop 1, on top of a's own hop-1 edge — none reachable from a alone
        # at that depth.
        result = {t.fact_id for t in chain_graph.bfs(["a", "d"], max_hops=1)}
        assert result == {"f1", "f3", "f4"}

    def test_unknown_seed_is_skipped_not_an_error(self, chain_graph):
        result = {t.fact_id for t in chain_graph.bfs(["nonexistent"], max_hops=2)}
        assert result == set()

    def test_max_facts_stops_early(self, chain_graph):
        result = chain_graph.bfs(["a"], max_hops=4, max_facts=2)
        assert len(result) == 2

    def test_zero_hops_is_rejected(self, chain_graph):
        with pytest.raises(ValueError, match="max_hops"):
            chain_graph.bfs(["a"], max_hops=0)


class TestShortestPath:
    def test_finds_the_direct_edge(self, chain_graph):
        path = chain_graph.shortest_path("a", "b")
        assert [t.fact_id for t in path] == ["f1"]

    def test_finds_a_multi_hop_path_in_order(self, chain_graph):
        path = chain_graph.shortest_path("a", "e")
        assert [t.fact_id for t in path] == ["f1", "f2", "f3", "f4"]

    def test_same_start_and_goal_is_an_empty_path(self, chain_graph):
        assert chain_graph.shortest_path("a", "a") == []

    def test_unreachable_goal_returns_none(self, chain_graph):
        assert chain_graph.shortest_path("a", "not-in-graph") is None

    def test_unknown_start_returns_none(self, chain_graph):
        assert chain_graph.shortest_path("not-in-graph", "a") is None

    def test_prefers_the_shorter_of_two_paths(self):
        # a-b-d (2 hops) vs a-c-d (2 hops) vs a direct a-d edge (1 hop):
        # the direct edge must win.
        g = KnowledgeGraph(
            {
                "f1": fact("a", "b"),
                "f2": fact("b", "d"),
                "f3": fact("a", "c"),
                "f4": fact("c", "d"),
                "f5": fact("a", "d"),
            }
        )
        path = g.shortest_path("a", "d")
        assert [t.fact_id for t in path] == ["f5"]


class TestFindEntities:
    def test_exact_label_match_is_case_insensitive(self, chain_graph):
        assert chain_graph.find_entities("B") == ["b"]

    def test_substring_match_when_no_exact_match(self):
        g = KnowledgeGraph({"f1": fact("x", "y", e1_label="a red fox", e2_label="a den")})
        assert "x" in g.find_entities("fox")

    def test_no_match_returns_empty(self, chain_graph):
        assert chain_graph.find_entities("nonexistent-thing") == []

    def test_blank_query_returns_empty(self, chain_graph):
        assert chain_graph.find_entities("   ") == []

    def test_respects_the_limit(self):
        # Five distinct entities all labeled with a shared substring.
        facts = {
            f"f{i}": fact(f"e{i}a", f"e{i}b", e1_label=f"item {i}", e2_label="other")
            for i in range(5)
        }
        g = KnowledgeGraph(facts)
        assert len(g.find_entities("item", limit=3)) == 3


class TestHeterogeneousEntityIds:
    def test_conceptnet_dbpedia_and_webchild_ids_coexist(self):
        # Real FVQA IDs differ wildly by source: ConceptNet URIs, DBpedia
        # URLs, and bare lowercase words for WebChild. The graph must treat
        # all three as opaque strings without merging unrelated entities
        # that happen to share a surface form.
        g = KnowledgeGraph(
            {
                "f1": fact("/c/en/dog/n", "/c/en/animal", kb="conceptnet"),
                "f2": fact(
                    "http://dbpedia.org/resource/Forest",
                    "http://dbpedia.org/resource/Category:Nature",
                    kb="dbpedia",
                ),
                "f3": fact("gove truck", "car", kb="webchild"),
            }
        )
        assert g.num_entities == 6
        assert g.has_entity("gove truck")
        assert g.has_entity("http://dbpedia.org/resource/Forest")
