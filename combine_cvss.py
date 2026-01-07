#!/usr/bin/env python3
"""
Combine Multiple SapiRecorder CSVs into One
Simply concatenates - preserves all original data
"""

import csv
import os
from pathlib import Path


def combine_csvs(input_files, output_csv="combined_session.csv"):
    """Simply concatenate CSVs - keep all original data unchanged."""
    
    total_events = 0
    
    with open(output_csv, 'w', newline='') as outfile:
        writer = None
        
        for csv_path in sorted(input_files):
            with open(csv_path, 'r') as infile:
                reader = csv.DictReader(infile)
                
                # Write header only once (first file)
                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                
                # Copy all rows as-is
                for row in reader:
                    writer.writerow(row)
                    total_events += 1
            
            print(f"  Added: {Path(csv_path).name}")
    
    print(f"\nCombined: {total_events} total events")
    print(f"Output: {output_csv}")


def combine_directory(input_dir, output_csv="combined_session.csv", pattern="*.csv"):
    """Combine all CSVs in a directory."""
    
    csv_files = sorted(Path(input_dir).glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0
    
    print(f"Found {len(csv_files)} CSV files\n")
    return combine_csvs([str(f) for f in csv_files], output_csv)


if __name__ == "__main__":
    # HARD-CODED PATHS
    INPUT_DIR = r"D:\V8\user0001"
    OUTPUT_CSV = r"D:\V8\combined_all_sessions.csv"
    
    print("="*60)
    print("CSV Combiner")
    print("="*60 + "\n")
    
    combine_directory(INPUT_DIR, OUTPUT_CSV, pattern="*.csv")
    
    print("="*60)