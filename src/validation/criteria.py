from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_gate_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Load validation-gate settings from YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gate configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"Gate configuration is empty or invalid: {config_path}"
        )

    required_sections = {
        "sample",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    }

    missing_sections = required_sections.difference(config)

    if missing_sections:
        raise ValueError(
            f"Gate configuration is missing sections: "
            f"{sorted(missing_sections)}"
        )

    return config