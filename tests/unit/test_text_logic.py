from jobnlp.harvest.policy import is_excluded_host, robots_sitemaps, tdmrep_reserved
from jobnlp.text.dedupe import UnionFind, normalised_hash
from jobnlp.text.geo import fold
from jobnlp.text.redact import _merge, plausible_name
from jobnlp.text.sections import classify_heading


def test_heading_rules() -> None:
    assert classify_heading("Ihr Profil:") == "requirements"
    assert classify_heading("Deine Aufgaben") == "tasks"
    assert classify_heading("What we offer") == "offer"
    assert classify_heading("Über uns") == "company"
    assert classify_heading("Lorem") is None


def test_union_find_merges_transitively() -> None:
    uf = UnionFind(4)
    uf.union(0, 2)
    uf.union(2, 3)
    assert [uf.find(i) for i in range(4)] == [0, 1, 0, 0]


def test_normalised_hash_ignores_case_and_punctuation() -> None:
    assert normalised_hash("Hallo, Welt!") == normalised_hash("hallo welt")


def test_tdmrep_reservation() -> None:
    assert tdmrep_reserved('[{"location": "/", "tdm-reservation": 1}]')
    assert not tdmrep_reserved('[{"location": "/", "tdm-reservation": 0}]')
    assert not tdmrep_reserved("not json")


def test_robots_sitemaps_and_host_exclusion() -> None:
    assert robots_sitemaps("User-agent: *\nSitemap: https://a.de/s.xml") == ["https://a.de/s.xml"]
    assert is_excluded_host("stellenanzeigen.de")
    assert not is_excluded_host("rossmann.de")


def test_fold_removes_umlauts() -> None:
    assert fold("Köln-Mülheim") == "koln mulheim"


def test_plausible_name_and_merge() -> None:
    assert plausible_name("Anna van Berg")
    assert not plausible_name("Waar")
    assert _merge([(0, 5), (3, 8), (10, 12)]) == [(10, 12), (0, 8)]
