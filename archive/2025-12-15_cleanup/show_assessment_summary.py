import pandas as pd

# Load the assessment report
df = pd.read_csv('daily_features_assessment_report.csv')

print("=" * 70)
print("📊 ASSESSMENT SUMMARY")
print("=" * 70)
print()

print(f"Total players in profile: {len(df)}")
print(f"✅ Successfully processed: {len(df[df['status'] == 'successful'])} ({len(df[df['status'] == 'successful'])*100//len(df)}%)")
print(f"❌ Failed (found in logs): {len(df[df['status'] == 'failed'])} ({len(df[df['status'] == 'failed'])*100//len(df)}%)")
print(f"⏳ Not processed: {len(df[df['status'] == 'not_processed'])} ({len(df[df['status'] == 'not_processed'])*100//len(df)}%)")
print()

# Check player 357119
player_357119 = df[df['player_id'] == 357119]
if not player_357119.empty:
    print("=" * 70)
    print("🎯 PLAYER 357119 STATUS:")
    print("=" * 70)
    print(player_357119.to_string())
    print()

# Show successful players stats
successful = df[df['status'] == 'successful']
if len(successful) > 0:
    print("=" * 70)
    print("✅ SUCCESSFUL PROCESSING STATS:")
    print("=" * 70)
    print(f"Average file size: {successful['file_size_mb'].mean():.2f} MB")
    print(f"Min file size: {successful['file_size_mb'].min():.2f} MB")
    print(f"Max file size: {successful['file_size_mb'].max():.2f} MB")
    print()

# Show sample of not processed
not_processed = df[df['status'] == 'not_processed']
if len(not_processed) > 0:
    print("=" * 70)
    print("⏳ NOT PROCESSED PLAYERS (Sample - first 20):")
    print("=" * 70)
    print(not_processed['player_id'].head(20).tolist())
    print()
    print(f"Note: Player 357119 is in this list and needs to be processed.")
    print(f"      The error 'cannot access local variable match_matches' likely")
    print(f"      occurred during processing, causing the script to fail before")
    print(f"      creating the output file.")
    print()

print("=" * 70)
print("📁 Generated Files:")
print("=" * 70)
print("  - daily_features_assessment_report.csv (full report)")
print("  - not_processed_players.csv (list of player IDs to process)")
if len(df[df['status'] == 'failed']) > 0:
    print("  - failed_players_to_reprocess.csv (list of failed player IDs)")
print("=" * 70)



