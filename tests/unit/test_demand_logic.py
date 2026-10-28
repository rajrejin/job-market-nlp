from jobnlp.gap.demand import demand_shares


def test_demand_among_nearest_postings() -> None:
    sets = {
        "a": frozenset({"python", "sql"}),
        "b": frozenset({"python", "sql", "docker"}),
        "c": frozenset({"excel"}),
    }
    # neighbours by Jaccard: a (1.0), b (0.67); c shares nothing and is ignored
    shares = demand_shares(["python", "sql"], sets, k=2)
    assert shares == {"python": 1.0, "sql": 1.0}
    assert demand_shares(["docker"], sets, k=5) == {"docker": 1.0}


def test_no_overlap_gives_empty() -> None:
    assert demand_shares(["rust"], {"a": frozenset({"go"})}) == {}
