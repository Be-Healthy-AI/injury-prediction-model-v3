import pandas as pd
import sys

file_path = r'production/deployments/England/challenger/Arsenal FC/daily_features/player_144028_daily_features.csv'

try:
    df = pd.read_csv(file_path, parse_dates=['date'], nrows=10)
    print("First 10 dates:")
    print(df[['date']].head(10))
    print(f"\nStart date: {df['date'].min()}")
    print(f"End date: {df['date'].max()}")
    print(f"Total rows: {len(pd.read_csv(file_path))}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
