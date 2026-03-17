#!/usr/bin/env python3
"""
Analyze server memory snapshots to detect leaks.

Usage:
    python scripts/analyze_server_snapshots.py <snapshot_dir>
"""

import sys
import tracemalloc
from pathlib import Path
from glob import glob


def analyze_snapshots(snapshot_dir: Path):
    """Analyze server memory snapshots for leaks."""
    
    # Find all server snapshots
    snapshot_files = sorted(glob(str(snapshot_dir / "server_snapshot_*.snapshot")))
    
    if len(snapshot_files) < 2:
        print("Need at least 2 snapshots for analysis")
        return
    
    print("="*70)
    print("SERVER MEMORY SNAPSHOT ANALYSIS")
    print("="*70)
    print(f"\nFound {len(snapshot_files)} snapshots")
    print()
    
    # Load all snapshots
    snapshots = []
    for f in snapshot_files:
        try:
            snap = tracemalloc.Snapshot.load(f)
            snapshots.append((Path(f).name, snap))
            print(f"  ✓ Loaded: {Path(f).name}")
        except Exception as e:
            print(f"  ✗ Failed to load {f}: {e}")
    
    if len(snapshots) < 2:
        print("\nNot enough valid snapshots")
        return
    
    print()
    print("="*70)
    print("MEMORY GROWTH ANALYSIS (First vs Last snapshot)")
    print("="*70)
    
    # Compare first and last
    first_name, first_snap = snapshots[0]
    last_name, last_snap = snapshots[-1]
    
    print(f"\nComparing:")
    print(f"  First: {first_name}")
    print(f"  Last:  {last_name}")
    print()
    
    top_stats = last_snap.compare_to(first_snap, 'lineno')
    
    print("Top memory growth:")
    print("-"*70)
    
    total_growth = 0
    growth_items = []
    
    for stat in top_stats:
        size_diff = stat.size_diff
        if size_diff != 0:
            growth_items.append((size_diff, stat))
            total_growth += size_diff
    
    # Sort by absolute size difference
    growth_items.sort(key=lambda x: abs(x[0]), reverse=True)
    
    # Show top 20
    for size_diff, stat in growth_items[:20]:
        sign = "+" if size_diff > 0 else ""
        size_kb = size_diff / 1024
        print(f"  {sign}{size_kb:.1f} KiB: {stat.traceback.format()[-1]}")
        if len(stat.traceback.format()) > 1:
            for line in stat.traceback.format()[-3:-1]:
                print(f"      {line}")
    
    print()
    print("-"*70)
    print(f"Total memory change: {total_growth / 1024:.1f} KiB")
    
    if total_growth > 1024 * 100:  # More than 100 KB growth
        print("⚠️  WARNING: Significant memory growth detected!")
    elif total_growth > 1024 * 10:  # More than 10 KB growth
        print("⚠️  NOTE: Some memory growth detected (may be normal)")
    else:
        print("✓ Memory usage is stable")
    
    # Show progression
    print()
    print("="*70)
    print("SNAPSHOT SIZE PROGRESSION")
    print("="*70)
    print()
    
    prev_size = None
    for name, snap in snapshots:
        total_size = sum(stat.size for stat in snap.statistics('lineno'))
        total_size_kb = total_size / 1024
        
        if prev_size is not None:
            diff = total_size - prev_size
            diff_str = f"({diff/1024:+.1f} KiB)"
        else:
            diff_str = "(baseline)"
        
        print(f"  {name}: {total_size_kb:.1f} KiB {diff_str}")
        prev_size = total_size
    
    # Detailed analysis of persistent objects
    print()
    print("="*70)
    print("POTENTIAL LEAK CANDIDATES (objects growing consistently)")
    print("="*70)
    print()
    
    # Track objects that grow in each snapshot
    growth_patterns = {}
    
    for i in range(1, len(snapshots)):
        prev_name, prev_snap = snapshots[i-1]
        curr_name, curr_snap = snapshots[i]
        
        diff = curr_snap.compare_to(prev_snap, 'lineno')
        
        for stat in diff:
            if stat.size_diff > 1024:  # Only significant growth
                key = str(stat.traceback)
                if key not in growth_patterns:
                    growth_patterns[key] = {
                        'stat': stat,
                        'count': 0,
                        'total_growth': 0,
                    }
                growth_patterns[key]['count'] += 1
                growth_patterns[key]['total_growth'] += stat.size_diff
    
    # Show patterns that appear in multiple snapshots
    persistent_leaks = [
        (info['total_growth'], info['count'], info['stat'])
        for key, info in growth_patterns.items()
        if info['count'] >= len(snapshots) // 3  # Appear in at least 1/3 of transitions
    ]
    
    persistent_leaks.sort(reverse=True)
    
    if persistent_leaks:
        for total_growth, count, stat in persistent_leaks[:10]:
            print(f"  Growth in {count}/{len(snapshots)-1} transitions, {total_growth/1024:.1f} KiB total:")
            for line in stat.traceback.format():
                print(f"    {line}")
            print()
    else:
        print("  No persistent growth patterns detected")
    
    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_server_snapshots.py <snapshot_dir>")
        sys.exit(1)
    
    snapshot_dir = Path(sys.argv[1])
    if not snapshot_dir.exists():
        print(f"Directory not found: {snapshot_dir}")
        sys.exit(1)
    
    analyze_snapshots(snapshot_dir)
