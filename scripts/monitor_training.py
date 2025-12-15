#!/usr/bin/env python3
"""
Monitor training script completion and notify when done
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import time
from datetime import datetime
from pathlib import Path

def monitor_training():
    """Monitor training completion"""
    results_file = Path('experiments/v4_baseline_metrics.json')
    summary_file = Path('experiments/v4_baseline_summary.md')
    models_dir = Path('models')
    
    print("="*80)
    print("TRAINING MONITOR")
    print("="*80)
    print(f"\nMonitoring for completion of: train_models_v4_baseline.py")
    print(f"Looking for: {results_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nChecking every 30 seconds...")
    print("="*80)
    
    check_count = 0
    start_time = time.time()
    
    while True:
        check_count += 1
        elapsed = time.time() - start_time
        
        # Check if results file exists
        if results_file.exists():
            print(f"\n{'='*80}")
            print("✅ TRAINING COMPLETED!")
            print(f"{'='*80}")
            print(f"\nCompletion detected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total monitoring time: {elapsed/60:.1f} minutes")
            print(f"Checks performed: {check_count}")
            
            # Check for all expected outputs
            print("\n📊 Checking output files:")
            print(f"   ✅ Metrics JSON: {results_file}")
            
            if summary_file.exists():
                print(f"   ✅ Summary Markdown: {summary_file}")
            else:
                print(f"   ⚠️  Summary Markdown: Not found")
            
            # Check for models
            expected_models = [
                'rf_model_v4_baseline.joblib',
                'gb_model_v4_baseline.joblib',
                'lr_model_v4_baseline.joblib'
            ]
            
            print("\n🤖 Checking trained models:")
            for model_file in expected_models:
                model_path = models_dir / model_file
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {model_file} ({size_mb:.2f} MB)")
                else:
                    print(f"   ⚠️  {model_file}: Not found")
            
            # Display summary
            if summary_file.exists():
                print(f"\n{'='*80}")
                print("PERFORMANCE SUMMARY")
                print(f"{'='*80}\n")
                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract and display the comparison table
                    lines = content.split('\n')
                    in_table = False
                    for i, line in enumerate(lines):
                        if '| Model | Dataset |' in line:
                            in_table = True
                        if in_table:
                            print(line)
                            if line.startswith('|') and '---' not in line and i < len(lines) - 1:
                                if not lines[i+1].startswith('|') and lines[i+1].strip() == '':
                                    break
            
            # Beep notification (Windows)
            if sys.platform == 'win32':
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 200)  # 1000 Hz for 200 ms
                    time.sleep(0.1)
            
            print(f"\n{'='*80}")
            print("✅ Monitoring complete. Training finished successfully!")
            print(f"{'='*80}\n")
            break
        
        # Status update every 2 minutes
        if check_count % 4 == 0:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Still training... ({elapsed/60:.1f} minutes elapsed)")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

