#!/usr/bin/env python3
"""
Generate summary report for production predictions.

Creates a Markdown report summarizing the prediction results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_PREDICTIONS_DIR = ROOT_DIR / "production_predictions" / "predictions"
PRODUCTION_REPORTS_DIR = ROOT_DIR / "production_predictions" / "reports"


def generate_report(predictions_file: Path, output_file: Path) -> None:
    """Generate markdown report from predictions file."""
    df = pd.read_csv(predictions_file)
    
    report_lines = [
        "# Production Injury Predictions Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Predictions Date:** {df['reference_date'].iloc[0] if not df.empty else 'N/A'}",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Players:** {df['player_id'].nunique()}",
        f"- **Total Predictions:** {len(df)}",
        "",
        "## Risk Level Distribution",
        "",
    ]
    
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        for level, count in risk_counts.items():
            pct = count / len(df) * 100
            report_lines.append(f"- **{level}:** {count} ({pct:.1f}%)")
    
    report_lines.extend([
        "",
        "## Probability Statistics",
        "",
    ])
    
    if 'ensemble_probability' in df.columns:
        report_lines.extend([
            f"- **Mean Probability:** {df['ensemble_probability'].mean():.3f}",
            f"- **Median Probability:** {df['ensemble_probability'].median():.3f}",
            f"- **Min Probability:** {df['ensemble_probability'].min():.3f}",
            f"- **Max Probability:** {df['ensemble_probability'].max():.3f}",
            f"- **Std Deviation:** {df['ensemble_probability'].std():.3f}",
        ])
    
    report_lines.extend([
        "",
        "## Top 10 Highest Risk Players",
        "",
        "| Player ID | Player Name | Probability | Risk Level |",
        "|-----------|------------|-------------|------------|",
    ])
    
    if 'ensemble_probability' in df.columns:
        top_risk = df.nlargest(10, 'ensemble_probability')
        for _, row in top_risk.iterrows():
            player_name = row.get('player_name', f'Player_{row["player_id"]}')
            report_lines.append(
                f"| {row['player_id']} | {player_name} | "
                f"{row['ensemble_probability']:.3f} | {row.get('risk_level', 'N/A')} |"
            )
    
    report_lines.extend([
        "",
        "## Model Comparison",
        "",
    ])
    
    if 'rf_probability' in df.columns and 'gb_probability' in df.columns:
        report_lines.extend([
            f"- **RF Mean:** {df['rf_probability'].mean():.3f}",
            f"- **GB Mean:** {df['gb_probability'].mean():.3f}",
            f"- **Ensemble Mean:** {df['ensemble_probability'].mean():.3f}",
        ])
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(report_lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--predictions-file',
        type=str,
        default=None,
        help='Path to combined predictions CSV file'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date string (YYYYMMDD) for predictions file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PRODUCTION_REPORTS_DIR),
        help='Directory for output report'
    )
    args = parser.parse_args()
    
    # Determine predictions file
    if args.predictions_file:
        predictions_file = Path(args.predictions_file)
    elif args.date:
        predictions_file = PRODUCTION_PREDICTIONS_DIR / f"predictions_{args.date}.csv"
    else:
        # Use most recent file
        prediction_files = list(PRODUCTION_PREDICTIONS_DIR.glob("predictions_*.csv"))
        if not prediction_files:
            print("❌ No predictions files found")
            return 1
        predictions_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return 1
    
    # Determine output file
    output_dir = Path(args.output_dir)
    date_str = args.date or predictions_file.stem.replace('predictions_', '')
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f"report_{date_str}.md"
    
    print("=" * 70)
    print("GENERATE PRODUCTION PREDICTIONS REPORT")
    print("=" * 70)
    print(f"📄 Input: {predictions_file}")
    print(f"📄 Output: {output_file}")
    print()
    
    generate_report(predictions_file, output_file)
    
    print(f"✅ Report generated: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

