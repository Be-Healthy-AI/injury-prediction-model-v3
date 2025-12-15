import pandas as pd
import os
from pathlib import Path

output_dir = Path('daily_features_output')
data_dir = Path('data_exports/transfermarkt/england/20251205')

print("=" * 70)
print("BATCH PROCESSING PROGRESS CHECK")
print("=" * 70)

# Count total players
if (data_dir / 'players_profile.csv').exists():
    players_df = pd.read_csv(data_dir / 'players_profile.csv', sep=';', encoding='utf-8')
    total_players = len(players_df['id'].unique())
    print(f"Total players to process: {total_players}")
else:
    print("ERROR: players_profile.csv not found")
    total_players = 0

# Count generated files
if output_dir.exists():
    generated_files = list(output_dir.glob('player_*_daily_features.csv'))
    successful = len(generated_files)
    
    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in generated_files) / (1024 * 1024)
    
    # Get file sizes for stats
    file_sizes = [f.stat().st_size / (1024 * 1024) for f in generated_files]
    avg_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
    
    print(f"Generated files: {successful}")
    print(f"Remaining: {total_players - successful}")
    print(f"Progress: {(successful * 100) // total_players if total_players > 0 else 0}%")
    print(f"\nTotal size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"Average file size: {avg_size:.2f} MB")
    
    if successful > 0:
        # Show some sample files
        print(f"\nSample generated files (first 10):")
        for f in sorted(generated_files)[:10]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.2f} MB)")
else:
    print("Output directory does not exist")
    successful = 0

print("=" * 70)



