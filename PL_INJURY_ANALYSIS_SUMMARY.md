# Premier League Injury Analysis Summary

## Overview
This analysis calculates Premier League injuries from seasons 2020/21 to 2024/25, filtering injuries to only count those that occurred when players were actually playing for Premier League clubs.

## Methodology

### 1. Data Sources
- **Injuries Data**: `injuries_data.csv` - Contains all injuries for players in the database
- **Career Data**: `players_career.csv` - Contains player transfer history and club affiliations
- **PL Club Lists**: Fetched from TransferMarkt for each season (2020-2024)

### 2. Key Steps

1. **Fetch PL Clubs**: For each season, the script fetches the list of Premier League clubs from TransferMarkt using the season ID parameter.

2. **Build Player Timeline**: Creates a timeline for each player showing which club they were at during each period based on transfer dates.

3. **Filter Injuries**: For each injury:
   - Checks if the injury occurred during a relevant season (2020/21 to 2024/25)
   - Determines which club the player was at on the injury date using the career timeline
   - Verifies if that club was in the Premier League during that season
   - Only counts injuries where the player was at a PL club

4. **Cost Estimation**: Estimates 2024/25 costs for Top 5 European leagues using:
   - Historical cost growth trends (average 25.8% per year)
   - Injury-to-cost ratio from previous seasons
   - Average of both methods for final estimate

## Results

### Injury Counts by Season
| Season | Injury Count | Historical Cost (M€) |
|--------|--------------|----------------------|
| 2020/21 | 543 | 376 |
| 2021/22 | 405 | 496 |
| 2022/23 | 432 | 696 |
| 2023/24 | 537 | 732 |
| 2024/25 | 656 | **907.55 (estimated)** |

### Key Findings
- **Total PL Injuries (2020/21-2024/25)**: 2,573 injuries
- **Average per Season**: 514.6 injuries
- **Year-over-Year Trend**: 
  - 2020/21: 543 injuries (baseline)
  - 2021/22: 405 injuries (-25.4%)
  - 2022/23: 432 injuries (+6.7%)
  - 2023/24: 537 injuries (+24.3%)
  - 2024/25: 656 injuries (+22.2%)

### Cost Estimation for Top 5 Leagues (2024/25)
**Estimated Cost: 907.55 million euros**

This estimate is based on:
- Historical cost growth rate: 25.8% per year
- Cost trend method: 920.88 M€
- Injury ratio method: 894.21 M€
- Final estimate: Average of both methods = **907.55 M€**

## Notes

1. **Club Name Matching**: The script uses fuzzy matching to handle variations in club names (e.g., "Manchester City FC" vs "Manchester City" vs "Man City").

2. **Data Limitations**: 
   - Injuries are only counted if the player's career timeline shows they were at a PL club on the injury date
   - Some injuries may be missed if career data is incomplete
   - The injury record's "clubs" field is used as a secondary check

3. **Cost Estimation Assumptions**:
   - Premier League represents a significant portion of Top 5 leagues spending
   - Historical growth trends continue
   - Cost per injury remains relatively stable

## Files Generated
- `calculate_pl_injuries.py` - Main analysis script
- `pl_injury_analysis_results.csv` - Results in CSV format
- `PL_INJURY_ANALYSIS_SUMMARY.md` - This summary document

## Running the Script
```bash
python calculate_pl_injuries.py
```

The script will:
1. Fetch PL clubs from TransferMarkt for each season
2. Load injuries and career data
3. Calculate PL injuries by season
4. Estimate 2024/25 costs
5. Save results to CSV
