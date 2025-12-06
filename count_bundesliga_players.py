import pandas as pd
import glob
import os

match_files = glob.glob("data_exports/transfermarkt/england/20251203/match_data/*.csv")
bundesliga_players = set()

print(f"Checking {len(match_files)} match files...")

for i, file in enumerate(match_files):
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(match_files)} files...")
    try:
        # Extract player_id from filename: match_<player_id>_<season>.csv
        filename = os.path.basename(file)
        player_id = int(filename.split('_')[1])
        
        # Read the CSV and check for Bundesliga (various formats)
        df = pd.read_csv(file)
        if "competition" in df.columns:
            # Check for Bundesliga in various formats
            if df["competition"].str.contains("bundesliga|1\. bundesliga|2\. bundesliga", 
                                             case=False, na=False, regex=True).any():
                bundesliga_players.add(player_id)
    except Exception as e:
        pass  # Skip errors silently

print(f"\nPlayers with Bundesliga matches: {len(bundesliga_players)}")
if bundesliga_players:
    print(f"Player IDs: {sorted(bundesliga_players)}")

