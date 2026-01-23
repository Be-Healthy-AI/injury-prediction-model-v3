import pandas as pd

file_path = r'production/raw_data/england/20260122/players_career.csv'

df = pd.read_csv(file_path, sep=';')
player = df[df['id'] == 144028]

print(f"Total career entries for player 144028: {len(player)}")
print("\nFirst 10 entries:")
if 'Date' in player.columns:
    print(player[['id', 'Date']].head(10))
    print(f"\nEarliest date: {player['Date'].min()}")
    print(f"Latest date: {player['Date'].max()}")
else:
    print("Available columns:", player.columns.tolist())
    print(player.head(10))
