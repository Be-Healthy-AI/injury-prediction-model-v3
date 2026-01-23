"""
Thin compatibility wrapper around the gold-standard daily features generator
used by the production lgbm_muscular_v1 model.

This module re-exports the core API from
models_production.lgbm_muscular_v1.code.daily_features.create_daily_features_v3
so that existing imports (scripts.create_daily_features_v3) continue to work.

We intentionally avoid any Windows-specific stdout/stderr wrapping or emoji
logging here; those concerns are handled by callers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import pandas as pd  # re-exported types for external code

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so models_production is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import the gold-standard implementation
from models_production.lgbm_muscular_v1.code.daily_features.create_daily_features_v3 import (  # type: ignore
    CONFIG,
    TEAM_COUNTRY_MAP,
    COMPETITION_TYPE_MAP,
    load_data_with_cache,
    preprocess_data_optimized,
    preprocess_career_data,
    initialize_team_country_map,
    initialize_competition_type_map,
    generate_daily_features_for_player_enhanced,
    generate_incremental_features_for_player,
)

__all__ = [
    "CONFIG",
    "TEAM_COUNTRY_MAP",
    "COMPETITION_TYPE_MAP",
    "load_data_with_cache",
    "preprocess_data_optimized",
    "preprocess_career_data",
    "initialize_team_country_map",
    "initialize_competition_type_map",
    "generate_daily_features_for_player_enhanced",
    "generate_incremental_features_for_player",
]


