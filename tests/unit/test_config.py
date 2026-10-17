from jobnlp.config import load_config


def test_config_seed_is_zero() -> None:
    assert load_config()["seed"] == 0
