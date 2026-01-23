# FBRef Rate Limiting Improvements

## ✅ Implemented Improvements

### 1. Enhanced Rate Limiting
- **Increased base delay**: 2.0s → 3.0s between requests
- **Random jitter**: Added 0-1 second random delay to avoid predictable patterns
- **Initial session delay**: 2 second delay before first request to let cloudscraper solve challenges

### 2. Exponential Backoff Retry Logic
- **Initial retry delay**: 5 seconds
- **Exponential backoff**: Delay doubles with each retry (5s → 10s → 20s → 40s → 60s)
- **Maximum delay cap**: 60 seconds
- **Random jitter**: Up to 20% additional random delay to avoid thundering herd

### 3. Special Handling for 403 Errors
- **Minimum wait**: 10 seconds minimum for 403 (rate limiting) errors
- **Increased retries**: 3 → 5 maximum retries
- **Better logging**: Clear messages indicating rate limiting vs other errors

### 4. Session Initialization
- **Pre-request delay**: 2 seconds before initial homepage visit
- **Post-request delay**: 1 second after initial request to ensure session is established

## Configuration

Settings are in `config/fbref_config.json`:

```json
{
  "rate_limit_seconds": 3.0,      // Base delay between requests
  "max_retries": 5,                // Maximum retry attempts
  "timeout_sec": 30                // Request timeout
}
```

Additional settings in code (can be added to config if needed):
- `initial_retry_delay`: 5.0 seconds
- `max_retry_delay`: 60.0 seconds
- `backoff_multiplier`: 2.0 (exponential)

## Retry Behavior

### Normal Errors (non-403)
- Attempt 1: Wait 5s + jitter
- Attempt 2: Wait 10s + jitter
- Attempt 3: Wait 20s + jitter
- Attempt 4: Wait 40s + jitter
- Attempt 5: Wait 60s + jitter (max)

### 403 Rate Limiting Errors
- Attempt 1: Wait 10s + jitter (minimum)
- Attempt 2: Wait 20s + jitter
- Attempt 3: Wait 40s + jitter
- Attempt 4: Wait 60s + jitter
- Attempt 5: Wait 60s + jitter (max)

## Expected Behavior

1. **First Request**: 2s delay → Homepage visit → 1s delay → Actual request
2. **Between Requests**: 3s + random jitter (0-1s)
3. **On Error**: Exponential backoff with jitter
4. **On 403**: Minimum 10s wait, then exponential backoff

## Monitoring

The scraper logs:
- Rate limiting delays (debug level)
- Retry attempts with wait times (warning level)
- Special messages for 403 errors

## Output Files Location

See `documentation/FBREF_OUTPUT_LOCATIONS.md` for complete details.

**Quick Reference**:
- **Mappings**: `data_exports/fbref/{country}/{YYYYMMDD}/players_mapping.csv`
- **Match Stats**: `data_exports/fbref/{country}/{YYYYMMDD}/match_stats/player_{fbref_id}_matches.csv`

Example:
```
data_exports/fbref/england/20251205/
├── players_mapping.csv
└── match_stats/
    ├── player_dc7f8a28_matches.csv
    └── ...
```

## Testing

To test the improved rate limiting:

```bash
python scripts/run_fbref_pipeline.py \
    --country england \
    --tm-data-dir "data_exports/transfermarkt/england/20251205" \
    --max-players 5 \
    --as-of-date 2025-12-05
```

The pipeline will now:
- Wait longer between requests
- Retry more times with exponential backoff
- Handle 403 errors more gracefully
- Show clear progress messages









