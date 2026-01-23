#!/usr/bin/env python3
"""Transform enriched metrics to match baseline format"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'

# Load enriched metrics (from enriched_comparison directory)
enriched_file = MODELS_DIR / 'enriched_comparison' / 'enriched_models_metrics.json'
if not enriched_file.exists():
    enriched_file = MODELS_DIR / 'lgbm_muscular_v4_enriched_metrics.json'

with open(enriched_file, 'r') as f:
    enriched_data = json.load(f)

# Transform to baseline format
baseline_format = {
    'model1_muscular': enriched_data.get('model1_lgbm_target1', {}),
    'model2_skeletal': enriched_data.get('model2_lgbm_target2', {}),
    'configuration': enriched_data.get('configuration', {})
}

# Update configuration
baseline_format['configuration']['version'] = 'v4_enriched'
baseline_format['configuration']['features'] = 'Layer 2 enriched (workload, recovery, injury history)'

# Save
output_file = MODELS_DIR / 'lgbm_muscular_v4_enriched_metrics.json'
with open(output_file, 'w') as f:
    json.dump(baseline_format, f, indent=2)

print(f"OK: Transformed and saved metrics to: {output_file}")
