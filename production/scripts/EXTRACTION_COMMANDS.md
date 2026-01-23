# Data Extraction Commands

## Saudi Arabia - Saudi Pro League Extraction

### Main Extraction Command

Extract all players from Saudi Pro League for seasons 2024/25 and 2025/26:

```bash
cd "C:\Users\joao.henriques\IPM V3"
python production/scripts/fetch_raw_data.py --country "Saudi Arabia" --league "Saudi Pro League" --competition-id "SA1" --competition-slug "saudi-pro-league" --seasons "2024,2025" --as-of-date 20260106
```

**Parameters:**
- `--country "Saudi Arabia"` → Creates folder `saudi_arabia` in `production/raw_data/`
- `--league "Saudi Pro League"` → Display name for logging
- `--competition-id "SA1"` → Transfermarkt competition ID (from URL: `wettbewerb/SA1`)
- `--competition-slug "saudi-pro-league"` → Transfermarkt URL slug
- `--seasons "2024,2025"` → Seasons 2024/25 and 2025/26
- `--as-of-date 20260106` → Reference date (YYYYMMDD format)

**Output Location:**
- `production/raw_data/saudi_arabia/20260106/` (main data folder)
- `production/raw_data/saudi_arabia/previous_seasons/` (older seasons match data)

### Verification Command

After extraction, verify match data completeness:

```bash
cd "C:\Users\joao.henriques\IPM V3"
python production/scripts/verify_spain_match_data.py --country "Saudi Arabia" --as-of-date 20260106 --league "Saudi Pro League"
```

**Note:** The script is now generic and works for any country/league.

### Fetch Missing Seasons Command

If verification finds missing seasons, fetch them:

```bash
cd "C:\Users\joao.henriques\IPM V3"
python production/scripts/fetch_missing_match_seasons.py --country "Saudi Arabia" --as-of-date 20260106 --league "Saudi Pro League"
```

**Note:** The script is now generic and works for any country/league.

---

## Spain - LaLiga Extraction (Reference)

For reference, the Spain extraction command:

```bash
cd "C:\Users\joao.henriques\IPM V3"
python production/scripts/fetch_raw_data.py --country Spain --league "LaLiga" --competition-id "ES1" --competition-slug "laliga" --seasons "2018,2019,2020,2021,2022,2023,2024,2025" --as-of-date 20260106
```

Verification:
```bash
python production/scripts/verify_spain_match_data.py --country Spain --as-of-date 20260106 --league "LaLiga"
```

Fetch missing:
```bash
python production/scripts/fetch_missing_match_seasons.py --country Spain --as-of-date 20260106 --league "LaLiga"
```
