# FBRef Scraping Notes

## 403 Forbidden and Fallback

FBRef often returns 403 for automated requests (Cloudflare/bot detection). The scraper handles this by:

1. **cloudscraper** – Used by default when installed (`pip install cloudscraper`).
2. **Playwright fallback** – If requests still get 403 after retries, the scraper automatically tries a headless Chromium browser:

```bash
pip install playwright
python -m playwright install chromium
```

No code changes needed: when 403 persists, the scraper logs "Trying Playwright fallback after 403..." and fetches the page with Playwright. Add `playwright>=1.40.0` to requirements for CI/production.

## Testing

- Harry Kane: `21a04d7d` – https://fbref.com/en/players/21a04d7d/
- Cole Palmer: `dc7f8a28` – https://fbref.com/en/players/dc7f8a28/









