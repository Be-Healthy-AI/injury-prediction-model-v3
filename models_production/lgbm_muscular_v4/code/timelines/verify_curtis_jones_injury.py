import pandas as pd
from pathlib import Path
from datetime import timedelta

# Check Curtis Jones's injury in the 2025/26 test dataset
timeline_file = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\timelines\test\timelines_35day_season_2025_2026_v4_muscular_test.csv")

if not timeline_file.exists():
    print(f"File not found: {timeline_file}")
    print("The file may not have been written due to permission error.")
    exit(1)

print("Loading timeline file...")
df = pd.read_csv(timeline_file, low_memory=False)
df['reference_date'] = pd.to_datetime(df['reference_date'])

print(f"Total timelines in file: {len(df):,}")

# Filter to Curtis Jones
player_id = 433188
player_timelines = df[df['player_id'] == player_id].copy()

print(f"\nCurtis Jones (ID: {player_id}) timelines: {len(player_timelines):,}")

# Check injury on 2025-10-26
injury_date = pd.Timestamp('2025-10-26')
expected_ref_dates = [injury_date - timedelta(days=d) for d in range(1, 6)]

print(f"\nChecking injury on {injury_date.date()}")
print(f"Expected reference dates: {[d.date() for d in expected_ref_dates]}")

found_correct = 0
found_incorrect = 0
missing = []

for ref_date in expected_ref_dates:
    matching = player_timelines[player_timelines['reference_date'].dt.normalize() == ref_date.normalize()]
    
    if len(matching) == 0:
        missing.append(ref_date)
        print(f"  [MISSING] {ref_date.date()}: No timeline found")
    else:
        target1_val = matching['target1'].iloc[0]
        target2_val = matching['target2'].iloc[0]
        
        # This should be a muscular injury, so target1 should be 1
        if target1_val == 1 and target2_val == 0:
            found_correct += 1
            print(f"  [OK] {ref_date.date()}: target1={target1_val}, target2={target2_val} (CORRECT)")
        else:
            found_incorrect += 1
            print(f"  [ERROR] {ref_date.date()}: target1={target1_val}, target2={target2_val} (INCORRECT - should be target1=1, target2=0)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Expected timelines: 5")
print(f"Correctly labeled (target1=1, target2=0): {found_correct}")
print(f"Incorrectly labeled: {found_incorrect}")
print(f"Missing: {len(missing)}")

if found_correct == 5:
    print("\n[SUCCESS] All 5 timelines for Curtis Jones's injury are correctly labeled!")
    print("The date parsing fix is working correctly!")
elif found_correct > 0:
    print(f"\n[PARTIAL] {found_correct}/5 timelines are correctly labeled.")
    print("The date parsing fix is partially working.")
else:
    print("\n[FAILURE] No timelines are correctly labeled.")
    print("The date parsing fix may not be working correctly.")
