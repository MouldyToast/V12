#!/usr/bin/env python3
"""
Merge Segments from Multiple Three-Dot Recording Sessions

Problem: Each session has segments/segment_0001.json, segment_0002.json, etc.
         Can't just copy to one folder - filenames conflict.

Solution: This script renames and copies all segments to a combined folder.

Usage:
    # Merge all sessions in a parent directory
    python merge_segments.py --input D:\V12\V12_Anchors_Continuous/three_dot_flow --output D:\V12\V12_Anchors_Continuous/combined_segments/
    
    # Merge specific session folders
    python merge_segments.py --sessions session_1/ session_2/ session_3/ --output combined/
    
    # Just list what would be merged (dry run)
    python merge_segments.py --input D:\V12\V12_Anchors_Continuous/three_dot_flow --dry-run

Output:
    combined_segments/
    ├── segment_0001.json   (was session_2024_01_01_100000/segments/segment_0001.json)
    ├── segment_0002.json   (was session_2024_01_01_100000/segments/segment_0002.json)
    ├── ...
    ├── segment_0201.json   (was session_2024_01_01_110000/segments/segment_0001.json)
    └── merge_manifest.json (tracks original sources)
"""

import json
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def find_session_folders(parent_dir: Path) -> list:
    """
    Find all session folders containing segments/ subdirectory.
    
    Looks for folders matching pattern: session_* or containing segments/
    """
    parent_dir = Path(parent_dir)
    session_folders = []
    
    # Look for session_* folders
    for folder in sorted(parent_dir.iterdir()):
        if folder.is_dir():
            segments_dir = folder / 'segments'
            if segments_dir.exists() and list(segments_dir.glob('segment_*.json')):
                session_folders.append(folder)
    
    return session_folders


def find_segment_files(session_folder: Path) -> list:
    """Find all segment JSON files in a session's segments/ folder."""
    segments_dir = session_folder / 'segments'
    if not segments_dir.exists():
        return []
    
    return sorted(segments_dir.glob('segment_*.json'))


def merge_segments(
    session_folders: list,
    output_dir: Path,
    dry_run: bool = False,
    update_segment_ids: bool = True
) -> dict:
    """
    Merge all segments from multiple sessions into one folder.
    
    Args:
        session_folders: List of session folder paths
        output_dir: Where to put combined segments
        dry_run: If True, just report what would happen
        update_segment_ids: Update segment_id in JSON to match new filename
    
    Returns:
        Manifest dict with merge information
    """
    output_dir = Path(output_dir)
    
    # Collect all segment files
    all_segments = []
    for session_folder in session_folders:
        session_name = session_folder.name
        segment_files = find_segment_files(session_folder)
        
        for seg_file in segment_files:
            all_segments.append({
                'session': session_name,
                'original_path': seg_file,
                'original_name': seg_file.name,
            })
    
    print(f"Found {len(all_segments)} segments across {len(session_folders)} sessions")
    
    if len(all_segments) == 0:
        print("No segments found!")
        return {}
    
    # Preview by session
    print("\nSegments per session:")
    session_counts = {}
    for seg in all_segments:
        session_counts[seg['session']] = session_counts.get(seg['session'], 0) + 1
    for session, count in sorted(session_counts.items()):
        print(f"  {session}: {count} segments")
    
    if dry_run:
        print("\n[DRY RUN] Would create:")
        print(f"  Output directory: {output_dir}")
        print(f"  Segments: segment_0001.json through segment_{len(all_segments):04d}.json")
        return {'dry_run': True, 'total_segments': len(all_segments)}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Manifest to track sources
    manifest = {
        'created': datetime.now().isoformat(),
        'total_segments': len(all_segments),
        'sessions': list(session_counts.keys()),
        'session_counts': session_counts,
        'segments': []
    }
    
    # Copy and rename segments
    print(f"\nCopying segments to {output_dir}/")
    
    for new_id, seg_info in enumerate(all_segments, start=1):
        new_filename = f"segment_{new_id:04d}.json"
        new_path = output_dir / new_filename
        
        # Load original segment
        with open(seg_info['original_path'], 'r') as f:
            segment_data = json.load(f)
        
        # Update segment_id if requested
        old_segment_id = segment_data.get('segment_id', '?')
        if update_segment_ids:
            segment_data['segment_id'] = new_id
            segment_data['_original_segment_id'] = old_segment_id
            segment_data['_original_session'] = seg_info['session']
        
        # Save to new location
        with open(new_path, 'w') as f:
            json.dump(segment_data, f, indent=2)
        
        # Track in manifest
        manifest['segments'].append({
            'new_id': new_id,
            'new_filename': new_filename,
            'original_session': seg_info['session'],
            'original_filename': seg_info['original_name'],
            'original_segment_id': old_segment_id,
        })
        
        # Progress indicator
        if new_id % 100 == 0:
            print(f"  Copied {new_id}/{len(all_segments)} segments...")
    
    print(f"\n✓ Merged {len(all_segments)} segments")
    print(f"  Output: {output_dir}")
    
    return manifest


def verify_merged_segments(output_dir: Path) -> bool:
    """Verify merged segments are valid."""
    output_dir = Path(output_dir)
    
    segment_files = sorted(output_dir.glob('segment_*.json'))
    print(f"\nVerifying {len(segment_files)} merged segments...")
    
    errors = []
    
    for i, seg_file in enumerate(segment_files, start=1):
        expected_name = f"segment_{i:04d}.json"
        
        # Check filename sequence
        if seg_file.name != expected_name:
            errors.append(f"Gap in sequence: expected {expected_name}, found {seg_file.name}")
        
        # Check file is valid JSON
        try:
            with open(seg_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            required = ['trajectory', 'AB', 'BC', 'turn']
            missing = [r for r in required if r not in data]
            if missing:
                errors.append(f"{seg_file.name}: missing fields {missing}")
                
        except json.JSONDecodeError as e:
            errors.append(f"{seg_file.name}: Invalid JSON - {e}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    
    print(f"✓ All {len(segment_files)} segments valid")
    return True


def print_stats(output_dir: Path):
    """Print statistics about merged segments."""
    output_dir = Path(output_dir)
    
    # Load manifest
    manifest_path = output_dir / 'merge_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        print(f"\n{'='*60}")
        print("MERGE STATISTICS")
        print(f"{'='*60}")
        print(f"Total segments: {manifest['total_segments']}")
        print(f"From {len(manifest['sessions'])} sessions:")
        for session, count in sorted(manifest['session_counts'].items()):
            print(f"  {session}: {count}")
    
    # Analyze segment content
    segment_files = list(output_dir.glob('segment_*.json'))
    if segment_files:
        turn_angles = []
        durations = []
        lengths = []
        
        for seg_file in segment_files[:100]:  # Sample first 100
            with open(seg_file, 'r') as f:
                data = json.load(f)
            turn_angles.append(data['turn']['angle_deg'])
            durations.append(data.get('duration_ms', 0))
            lengths.append(data['trajectory']['length'])
        
        print(f"\nContent sample (first {len(turn_angles)} segments):")
        print(f"  Turn angles: mean={sum(turn_angles)/len(turn_angles):.1f}°")
        print(f"  Durations:   mean={sum(durations)/len(durations):.0f} ms")
        print(f"  Lengths:     mean={sum(lengths)/len(lengths):.0f} points")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge segments from multiple three-dot recording sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge all sessions in a parent directory
    python merge_segments.py --input D:/V12/three_dot_flow/ --output D:/V12/combined_segments/
    
    # Merge specific session folders  
    python merge_segments.py --sessions session_1/ session_2/ --output combined/
    
    # Dry run - just show what would happen
    python merge_segments.py --input D:/V12/three_dot_flow/ --dry-run
    
    # Verify existing merged segments
    python merge_segments.py --verify D:/V12/combined_segments/
"""
    )
    
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Parent directory containing session folders')
    parser.add_argument('--sessions', nargs='+', type=str, default=None,
                        help='Specific session folders to merge')
    parser.add_argument('--output', '-o', type=str, default='combined_segments',
                        help='Output directory for merged segments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without copying')
    parser.add_argument('--verify', type=str, default=None,
                        help='Verify existing merged segments directory')
    parser.add_argument('--stats', type=str, default=None,
                        help='Show statistics for merged segments directory')
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        verify_merged_segments(Path(args.verify))
        print_stats(Path(args.verify))
        exit(0)
    
    # Stats mode
    if args.stats:
        print_stats(Path(args.stats))
        exit(0)
    
    # Determine session folders
    if args.sessions:
        session_folders = [Path(s) for s in args.sessions]
    elif args.input:
        session_folders = find_session_folders(Path(args.input))
    else:
        # Default: look in current directory
        session_folders = find_session_folders(Path('.'))
    
    if not session_folders:
        print("No session folders found!")
        print("Use --input to specify parent directory or --sessions for specific folders")
        exit(1)
    
    print(f"Found {len(session_folders)} session folders:")
    for sf in session_folders:
        print(f"  {sf}")
    
    # Merge
    manifest = merge_segments(
        session_folders,
        Path(args.output),
        dry_run=args.dry_run
    )
    
    # Verify if not dry run
    if not args.dry_run and manifest:
        verify_merged_segments(Path(args.output))
        print_stats(Path(args.output))
