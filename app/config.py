from __future__ import annotations

"""Configuration utilities.

Handles loading configuration from `.env` files.
This keeps secrets out of the environment per user instructions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from .errors import ConfigError


@dataclass
class AppConfig:
    """Application configuration."""

    hf_token: Optional[str] = None


def load_config(env_path: Path) -> AppConfig:
    """Load configuration from a `.env` file.

    Parameters
    ----------
    env_path:
        Path to `.env` file containing `HF_TOKEN`.

    Returns
    -------
    AppConfig
        Parsed configuration.

    Raises
    ------
    ConfigError
        If the file is missing or unreadable.
    """

    if not env_path.exists():
        raise ConfigError(f"Config file {env_path} not found")
    data = dotenv_values(env_path)
    token = data.get("HF_TOKEN")
    if token:
        token = token.strip()
    return AppConfig(hf_token=token or None)
