# FBRef Output Files Location

## Output Directory Structure

When you run the FBRef pipeline, data is saved to:

```
data_exports/fbref/
└── {country}/
    └── {YYYYMMDD}/              # Date folder (e.g., 20251205)
        ├── players_mapping.csv  # TransferMarkt ↔ FBRef player mappings
        └── match_stats/         # Match statistics per player
            ├── player_{fbref_id}_matches.csv
            ├── player_{fbref_id}_matches.csv
            └── ...
```

## Example Paths

### For England, December 5, 2025:
```
data_exports/fbref/england/20251205/
├── players_mapping.csv
└── match_stats/
    ├── player_dc7f8a28_matches.csv  # Cole Palmer
    ├── player_abc123_matches.csv
    └── ...
```

## File Descriptions

### 1. `players_mapping.csv`
**Location**: `data_exports/fbref/{country}/{YYYYMMDD}/players_mapping.csv`

**Columns**:
- `transfermarkt_id`: TransferMarkt player ID
- `transfermarkt_name`: Player name from TransferMarkt
- `fbref_id`: FBRef player ID (e.g., 'dc7f8a28')
- `fbref_name`: Player name from FBRef
- `fbref_url`: Full FBRef profile URL
- `date_of_birth`: Date of birth (for validation)
- `match_confidence`: Matching confidence (0.0-1.0)
- `match_method`: How the match was found ('exact_name', 'fuzzy_name', 'manual', etc.)
- `last_verified`: Last time mapping was validated
- `is_active`: Whether mapping is still valid (True/False)

**Purpose**: Maps TransferMarkt players to FBRef players for future reference.

### 2. `player_{fbref_id}_matches.csv`
**Location**: `data_exports/fbref/{country}/{YYYYMMDD}/match_stats/player_{fbref_id}_matches.csv`

**Columns**: See `scripts/data_collection/fbref_transformers.py` for full schema.

**Key columns**:
- `fbref_player_id`: FBRef player ID
- `match_date`: Match date
- `season`: Season (e.g., '2024-2025')
- `competition`: Competition name
- `team`: Player's team
- `opponent`: Opposing team
- `goals`, `assists`, `minutes`: Basic stats
- `pass_accuracy_pct`, `xG`, `touches`, etc.: Advanced stats

**Purpose**: Detailed match-by-match statistics for each player.

## How to Access Output Files

### From Command Line
```bash
# View mapping file
cat data_exports/fbref/england/20251205/players_mapping.csv

# List all match stats files
ls data_exports/fbref/england/20251205/match_stats/

# View a specific player's matches
cat data_exports/fbref/england/20251205/match_stats/player_dc7f8a28_matches.csv
```

### From Python
```python
import pandas as pd
from pathlib import Path

# Load mappings
mappings = pd.read_csv('data_exports/fbref/england/20251205/players_mapping.csv')

# Load a player's match stats
player_matches = pd.read_csv(
    'data_exports/fbref/england/20251205/match_stats/player_dc7f8a28_matches.csv',
    parse_dates=['match_date']
)
```

## File Naming Convention

- **Mapping file**: Always named `players_mapping.csv`
- **Match stats**: Named `player_{fbref_id}_matches.csv`
  - Example: `player_dc7f8a28_matches.csv` for Cole Palmer

## Notes

- Files are saved in UTF-8 encoding with BOM (for Excel compatibility)
- Match statistics are stored per player (one file per player)
- The mapping file is cumulative (appends new mappings)
- If a player's match stats already exist, the pipeline will skip scraping (unless `--force-remap` is used)









