from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "configuration.yaml"
)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Некорректный формат configuration.yaml")

    return config