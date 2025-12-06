"""
Diagnose why stats columns are not being populated.
Shows the actual table structure and column mapping.
"""
import sys
sys.path.insert(0, 'scripts/data_collection')
from transfermarkt_scraper import TransfermarktScraper
from transformers import transform_matches
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

def diagnose_player(player_id: int, season: int = 2024):
    """Diagnose column mapping for a specific player."""
    scraper = TransfermarktScraper()
    
    print(f"Diagnosing player {player_id}, season {season}")
    print("="*70)
    
    # Fetch the HTML directly to see structure
    url = f"{scraper.config.base_url}/spieler/leistungsdatendetails/spieler/{player_id}/saison/{season}/verein/0/liga/0/wettbewerb//pos/0/trainer_id/0/plus/1"
    soup = scraper._fetch_soup(url)
    
    # Find match tables
    all_tables = soup.select('table')
    for i, table in enumerate(all_tables):
        headers = table.select('thead th')
        if headers:
            header_texts = [h.get_text(strip=True) for h in headers]
            if any(keyword in ' '.join(header_texts).lower() for keyword in ['matchday', 'date', 'venue', 'opponent', 'result']):
                print(f"\n📋 TABLE {i} STRUCTURE:")
                print("-"*70)
                
                # Show HTML headers
                print("\nHTML Headers (with titles):")
                for j, h in enumerate(headers):
                    text = h.get_text(strip=True)
                    title = h.get('title', '').strip()
                    print(f"  {j:2d}: text='{text:15s}' title='{title}'")
                
                # Parse with pandas
                dfs = pd.read_html(StringIO(str(table)), flavor="bs4")
                if dfs:
                    df = dfs[0]
                    print(f"\nPandas DataFrame columns ({len(df.columns)}):")
                    for j, col in enumerate(df.columns):
                        print(f"  {j:2d}: {repr(col)}")
                    
                    # Show first row data
                    if len(df) > 0:
                        print(f"\nFirst row - all non-empty values:")
                        for j, col in enumerate(df.columns):
                            val = df[col].iloc[0]
                            if pd.notna(val) and str(val) != 'nan' and str(val).strip() and str(val) != '-':
                                print(f"  Col {j:2d} ({col:25s}): {repr(val)}")
                
                # Now test the scraper method
                print(f"\n🔍 TESTING SCRAPER METHOD:")
                print("-"*70)
                matches_raw = scraper.fetch_player_match_log(None, player_id, season=season)
                
                print(f"\nRaw DataFrame from scraper ({len(matches_raw)} rows, {len(matches_raw.columns)} cols):")
                print(f"Columns: {matches_raw.columns.tolist()[:15]}...")
                
                # Check for stats columns
                print(f"\nStats-related columns found:")
                stats_found = []
                for col in matches_raw.columns:
                    if any(x in str(col).lower() for x in ['position', 'goal', 'assist', 'pos', 'yellow', 'red', 'sub', 'minute']):
                        stats_found.append(col)
                        populated = matches_raw[col].notna().sum() if len(matches_raw) > 0 else 0
                        sample = matches_raw[col].head(3).tolist() if len(matches_raw) > 0 else []
                        print(f"  {repr(col):30s}: {populated}/{len(matches_raw)} populated, sample: {sample}")
                
                if not stats_found:
                    print("  ❌ No stats columns found!")
                
                # Check after transformation
                print(f"\n🔄 AFTER TRANSFORMATION:")
                print("-"*70)
                matches_transformed = transform_matches(player_id, "Test Player", matches_raw)
                
                stats = ['position', 'goals', 'assists', 'own_goals', 'yellow_cards', 
                        'second_yellow_cards', 'red_cards', 'substitutions_on', 
                        'substitutions_off', 'minutes_played']
                
                print(f"Transformed columns: {matches_transformed.columns.tolist()}")
                print(f"\nStats columns population:")
                for col in stats:
                    if col in matches_transformed.columns:
                        populated = matches_transformed[col].notna().sum()
                        pct = 100 * populated / len(matches_transformed) if len(matches_transformed) > 0 else 0
                        status = "✅" if populated > 0 else "❌"
                        print(f"  {status} {col:25s}: {populated:4d}/{len(matches_transformed):4d} ({pct:5.1f}%)")
                        if populated > 0:
                            sample = matches_transformed[col].dropna().head(3).tolist()
                            print(f"      Sample values: {sample}")
                    else:
                        print(f"  ❌ {col:25s}: Column not found")
                
                break  # Just check first match table

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose match column mapping')
    parser.add_argument('--player-id', type=int, default=158863,
                       help='Player ID to diagnose (default: 158863 - Iñigo Martínez)')
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year (default: 2024)')
    
    args = parser.parse_args()
    diagnose_player(args.player_id, args.season)


