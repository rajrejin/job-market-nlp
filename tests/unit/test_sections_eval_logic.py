from jobnlp.text.sections_eval import line_sections, score


def test_line_sections_propagate() -> None:
    assert line_sections(4, {1: "tasks", 3: "offer"}) == ["company", "tasks", "tasks", "offer"]


def test_score_hand_computed() -> None:
    # line 1 is a heading the splitter knows; line 3 a heading it mislabels as requirements
    text = "Wir sind toll\nIhre Aufgaben:\n- planen\nIhr Profil:\n- SQL"
    perfect = score([{"text": text, "labels": {"0": "none", "1": "tasks", "3": "requirements"}}])
    assert perfect["heading_accuracy"] == 1.0 and perfect["boundary_precision"] == 1.0
    assert perfect["line_accuracy"] == 1.0
    wrong = score([{"text": text, "labels": {"0": "none", "1": "tasks", "3": "offer"}}])
    assert wrong["heading_accuracy"] == 0.5 and wrong["boundary_recall"] == 1.0
    assert wrong["line_accuracy"] == 3 / 5
