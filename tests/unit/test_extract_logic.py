from jobnlp.skills.extract import keep_alt_label
from jobnlp.skills.mine import clean_item


def test_single_word_alt_labels_only_when_tool_like() -> None:
    assert keep_alt_label("SQL") and keep_alt_label("JavaScript") and keep_alt_label("C++")
    assert keep_alt_label(".NET") and keep_alt_label("Node.js")
    assert keep_alt_label("machine learning")  # multi-word labels are always kept
    assert not keep_alt_label("data") and not keep_alt_label("design")


def test_mined_names_stop_at_function_words() -> None:
    assert clean_item("Deutschkenntnisse in Wort") == "Deutschkenntnisse"
    assert clean_item("Erfahrung mit Docker") == "Docker"  # leading frame words dropped
    assert clean_item("Wirtschafts-") is None
    assert clean_item("C1") is None
