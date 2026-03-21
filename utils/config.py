import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    local_path = CONFIG_PATH.with_name("settings_local.yaml")
    if local_path.exists():
        with open(local_path, "r") as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local_cfg)

    odds_key = os.environ.get("ODDS_API_KEY", "")
    if odds_key:
        cfg["odds_api"]["api_key"] = odds_key

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
