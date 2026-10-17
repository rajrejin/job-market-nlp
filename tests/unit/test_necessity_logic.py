from jobnlp.skills.necessity import necessity_at


def _nec(text: str, word: str) -> str:
    return necessity_at(text, text.index(word))


def test_line_cue_marks_nice() -> None:
    assert _nec("- SQL-Kenntnisse sind von Vorteil", "SQL") == "nice"


def test_negated_must_cue_is_nice() -> None:
    text = "- Erfahrung in Buchhaltung ist von Vorteil, aber nicht zwingend erforderlich"
    assert _nec(text, "Buchhaltung") == "nice"


def test_heading_scope() -> None:
    text = "Must have:\n- Python\nNice to have:\n- Docker"
    assert _nec(text, "Python") == "must"
    assert _nec(text, "Docker") == "nice"


def test_default_is_must() -> None:
    assert _nec("- Erfahrung mit Python", "Python") == "must"


def test_cue_in_other_sentence_does_not_leak() -> None:
    text = "- Sehr gute Python-Kenntnisse. Docker ist ein Plus."
    assert _nec(text, "Python") == "must"
    assert _nec(text, "Docker") == "nice"
