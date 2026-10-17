from jobnlp.gap.guards import check_facts, check_no_new_skills, check_skill_in_gap


def _keys(text: str) -> set[str]:
    # toy skill detector for testing guard logic: capitalised words are skills
    return {w for w in text.replace(",", " ").split() if w[:1].isupper()}


def test_skill_must_come_from_gap() -> None:
    assert check_skill_in_gap("sql", {"sql", "python"}).passed
    assert not check_skill_in_gap("java", {"sql"}).passed


def test_new_skill_is_flagged_unless_supported() -> None:
    bullet = "built reports with Excel"
    assert not check_no_new_skills(
        "built Tableau reports", bullet, set(), "Tableau", False, _keys
    ).passed
    assert check_no_new_skills(
        "built Tableau reports", bullet, set(), "Tableau", True, _keys
    ).passed
    assert check_no_new_skills(
        "built Tableau reports", bullet, {"Tableau"}, "X", False, _keys
    ).passed


def test_numbers_must_exist_in_original() -> None:
    assert check_facts("cut costs by 20 %", "cut costs by 20%").passed
    report = check_facts("cut costs by 35% in 2021", "cut costs by 20%")
    assert not report.passed and len(report.violations) == 2


def test_organisations_must_exist_in_original() -> None:
    orgs = lambda s: {"Siemens"} if "Siemens" in s else set()  # noqa: E731
    assert not check_facts("worked at Siemens", "worked at a large firm", orgs).passed
