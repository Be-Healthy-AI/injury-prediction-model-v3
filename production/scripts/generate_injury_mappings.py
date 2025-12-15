#!/usr/bin/env python3
"""
Generate comprehensive injury mappings from existing injuries data.

Analyzes the injuries_data.csv file to create mappings for:
- injury_class (muscular, skeletal, other, unknown, no_injury)
- body_part (lower_leg, knee, upper_leg, hip, upper_body, head, illness, or empty)

Output: production/config/injury_mappings.json
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, Any

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

def generate_injury_mappings(injuries_file: Path) -> Dict[str, Dict[str, str]]:
    """
    Generate comprehensive injury mappings from existing data.
    
    Returns a dictionary mapping injury_type -> {injury_class, body_part}
    """
    print(f"Reading injuries data from: {injuries_file}")
    df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
    
    print(f"Total injuries: {len(df)}")
    print(f"Unique injury types: {df['injury_type'].nunique()}")
    
    # Group by injury_type and get the most common classification
    mappings = {}
    
    for injury_type in df['injury_type'].unique():
        if pd.isna(injury_type):
            continue
            
        injury_data = df[df['injury_type'] == injury_type]
        
        # Get most common injury_class (excluding empty/NaN)
        injury_classes = injury_data['injury_class'].dropna()
        if len(injury_classes) > 0:
            most_common_class = injury_classes.mode()[0] if len(injury_classes.mode()) > 0 else 'unknown'
        else:
            most_common_class = 'unknown'
        
        # Get most common body_part (excluding empty/NaN)
        body_parts = injury_data['body_part'].dropna()
        if len(body_parts) > 0:
            most_common_body_part = body_parts.mode()[0] if len(body_parts.mode()) > 0 else ''
        else:
            most_common_body_part = ''
        
        mappings[str(injury_type)] = {
            'injury_class': most_common_class,
            'body_part': most_common_body_part
        }
    
    return mappings

def main():
    # Find the reference injuries file
    reference_file = PRODUCTION_ROOT / "raw_data" / "england" / "20251205" / "injuries_data.csv"
    
    if not reference_file.exists():
        print(f"[ERROR] Reference file not found: {reference_file}")
        return 1
    
    print("=" * 70)
    print("GENERATING INJURY MAPPINGS")
    print("=" * 70)
    print()
    
    # Generate mappings
    mappings = generate_injury_mappings(reference_file)
    
    # Output file
    output_file = PRODUCTION_ROOT / "config" / "injury_mappings.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mappings
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Generated {len(mappings)} injury type mappings")
    print(f"[OK] Saved to: {output_file}")
    print()
    
    # Show summary
    class_counts = Counter(m['injury_class'] for m in mappings.values())
    body_part_counts = Counter(m['body_part'] for m in mappings.values() if m['body_part'])
    
    print("Summary:")
    print(f"  Injury classes: {dict(class_counts)}")
    print(f"  Body parts: {dict(body_part_counts)}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

