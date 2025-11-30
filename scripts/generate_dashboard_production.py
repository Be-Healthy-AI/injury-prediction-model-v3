#!/usr/bin/env python3
"""
Generate interactive HTML dashboard for production predictions.

Creates a dashboard with visualizations of prediction results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_PREDICTIONS_DIR = ROOT_DIR / "production_predictions" / "predictions"
PRODUCTION_DASHBOARDS_DIR = ROOT_DIR / "production_predictions" / "dashboards"


def generate_dashboard(predictions_file: Path, output_file: Path) -> None:
    """Generate HTML dashboard from predictions file."""
    df = pd.read_csv(predictions_file)
    
    # Generate HTML with embedded JavaScript for charts
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Injury Predictions Dashboard - {df['reference_date'].iloc[0] if not df.empty else 'N/A'}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Injury Predictions Dashboard</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Predictions Date:</strong> {df['reference_date'].iloc[0] if not df.empty else 'N/A'}</p>
        
        <div class="summary">
            <div class="stat-card">
                <h3>{df['player_id'].nunique()}</h3>
                <p>Total Players</p>
            </div>
            <div class="stat-card">
                <h3>{len(df)}</h3>
                <p>Total Predictions</p>
            </div>
"""
    
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        for level, count in risk_counts.items():
            html_content += f"""
            <div class="stat-card">
                <h3>{count}</h3>
                <p>{level} Risk</p>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="chart-container">
            <h2>Risk Level Distribution</h2>
            <div id="riskChart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Probability Distribution</h2>
            <div id="probChart"></div>
        </div>
        
        <h2>Top 10 Highest Risk Players</h2>
        <table>
            <thead>
                <tr>
                    <th>Player ID</th>
                    <th>Player Name</th>
                    <th>Probability</th>
                    <th>Risk Level</th>
                </tr>
            </thead>
            <tbody>
"""
    
    if 'ensemble_probability' in df.columns:
        top_risk = df.nlargest(10, 'ensemble_probability')
        for _, row in top_risk.iterrows():
            player_name = row.get('player_name', f'Player_{row["player_id"]}')
            html_content += f"""
                <tr>
                    <td>{row['player_id']}</td>
                    <td>{player_name}</td>
                    <td>{row['ensemble_probability']:.3f}</td>
                    <td>{row.get('risk_level', 'N/A')}</td>
                </tr>
"""
    
    # Risk level pie chart data
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        risk_labels = risk_counts.index.tolist()
        risk_values = risk_counts.values.tolist()
    else:
        risk_labels = []
        risk_values = []
    
    # Probability histogram data
    if 'ensemble_probability' in df.columns:
        prob_data = df['ensemble_probability'].tolist()
    else:
        prob_data = []
    
    html_content += f"""
            </tbody>
        </table>
    </div>
    
    <script>
        // Risk Level Distribution Chart
        var riskData = [{{
            values: {risk_values},
            labels: {risk_labels},
            type: 'pie',
            marker: {{
                colors: ['#4CAF50', '#FF9800', '#F44336']
            }}
        }}];
        
        var riskLayout = {{
            title: 'Risk Level Distribution',
            height: 400
        }};
        
        Plotly.newPlot('riskChart', riskData, riskLayout);
        
        // Probability Distribution Chart
        var probData = [{{
            x: {prob_data},
            type: 'histogram',
            marker: {{
                color: '#667eea'
            }}
        }}];
        
        var probLayout = {{
            title: 'Injury Probability Distribution',
            xaxis: {{title: 'Probability'}},
            yaxis: {{title: 'Count'}},
            height: 400
        }};
        
        Plotly.newPlot('probChart', probData, probLayout);
    </script>
</body>
</html>
"""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')


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
        default=str(PRODUCTION_DASHBOARDS_DIR),
        help='Directory for output dashboard'
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
    output_file = output_dir / f"dashboard_{date_str}.html"
    
    print("=" * 70)
    print("GENERATE PRODUCTION PREDICTIONS DASHBOARD")
    print("=" * 70)
    print(f"📄 Input: {predictions_file}")
    print(f"📄 Output: {output_file}")
    print()
    
    generate_dashboard(predictions_file, output_file)
    
    print(f"✅ Dashboard generated: {output_file}")
    print(f"   Open in browser to view interactive charts")
    
    return 0


if __name__ == "__main__":
    exit(main())

