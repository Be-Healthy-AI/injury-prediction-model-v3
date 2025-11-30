# Transfermarkt Reference Schemas

Source workbooks live in `original_data/` with `YYYYMMDD_<dataset>.xlsx` naming. The tables below describe the current schema contract that the automated pipeline must reproduce exactly.

## 1. Players Profile (`20251106_players_profile.xlsx`, sheet `profiles`)

| Column | Type | Notes |
| --- | --- | --- |
| `id` | int64 | Transfermarkt player ID (primary key) |
| `name` | string | Localized player name, includes diacritics |
| `position` | string | Primary position label as shown on profile |
| `date_of_birth` | datetime | Parsed date, timezone-na |
| `nationality1` | string | Primary nationality; empty if unavailable |
| `nationality2` | string | Secondary nationality or blank |
| `height` | float | Height in cm, NaN when missing |
| `foot` | string | Dominant foot (`right`, `left`, `both`, or blank) |
| `joined_on` | datetime | Date of arrival at current club |
| `signed_from` | string | Source club string as displayed |

## 2. Players Career (`20251106_players_career.xlsx`, sheet `careers`)

| Column | Type | Notes |
| --- | --- | --- |
| `id` | int64 | Transfermarkt player ID |
| `name` | string | Player name |
| `Season` | string | Season span formatted like `22/23` |
| `Date` | datetime | Transfer date |
| `From` | string | Origin club label |
| `To` | string | Destination club label |
| `VM` | int64 | Market value at time of move (euros) |
| `Value` | string | Transfer fee text (may contain `-`) |

## 3. Injuries (`20251106_injuries_data.xlsx`, sheet `injuries`)

| Column | Type | Notes |
| --- | --- | --- |
| `player_id` | int64 | Transfermarkt player ID |
| `season` | string | Season span (e.g., `09/10`) |
| `injury_type` | string | Free-text injury description |
| `fromDate` | datetime | Start date of injury |
| `untilDate` | datetime | End date of injury; may be NaT when ongoing |
| `days` | float | Duration in days |
| `clubs` | string | Club affected |
| `lost_games` | float | Number of games missed (NaN allowed) |
| `no_physio_injury` | float | Indicator (1.0 when physio data missing) |

## 4. Matches (`20251106_match_data.xlsx`, sheet `matches`)

| Column | Type | Notes |
| --- | --- | --- |
| `season` | string | Season formatted like `2000-2001` |
| `player_id` | int64 | Transfermarkt player ID |
| `player_name` | string | Player name |
| `competition` | string | Competition label |
| `journey` | string | Matchday (text/number) |
| `date` | datetime | Match date |
| `home_team` | string | Home club |
| `away_team` | string | Away club |
| `result` | string | Score result |
| `position` | string | Position played |
| `goals` | float | Goals scored |
| `assists` | float | Assists |
| `own_goals` | float | Own goals |
| `yellow_cards` | float | Yellow cards |
| `second_yellow_cards` | float | Second yellows |
| `red_cards` | float | Red cards |
| `substitutions_on` | float | Minute substituted on |
| `substitutions_off` | float | Minute substituted off |
| `minutes_played` | string | Minute string (e.g., `90'`) |
| `transfermarkt_score` | float | TM rating (NaN when unavailable) |

## 5. Teams (`20251106_teams_data.xlsx`, sheet `teams`)

| Column | Type | Notes |
| --- | --- | --- |
| `team` | string | Team name |
| `country` | string | Country text |

## 6. Competitions (`20251106_competition_data.xlsx`, sheet `competitions`)

| Column | Type | Notes |
| --- | --- | --- |
| `competition` | string | Competition name |
| `country` | string | Country hosting competition |
| `type` | string | Category such as `Main League`, `Cup`, etc. |

These schemas should be kept up to date whenever Transfermarkt changes the site structure or when additional derived columns are introduced. Any pipeline output must match these field names, data types, and nullability constraints before it is accepted.

