from pathlib import Path
import pandas as pd

match_file = Path('production/raw_data/spain/previous_seasons/match_251892_2024_2025.csv')
print('File exists:', match_file.exists())

if match_file.exists():
    df = pd.read_csv(match_file, encoding='utf-8-sig', nrows=5)
    print('\nColumns:', list(df.columns))
    print('\nSample data:')
    print(df.head())
    if 'club' in df.columns:
        print('\nUnique clubs in sample:')
        print(df['club'].unique())
    elif 'team' in df.columns:
        print('\nUnique teams in sample:')
        print(df['team'].unique())

