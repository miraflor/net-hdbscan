"""Read-only publication verification benchmark for existing outputs.

Run after generating a realistically sized grouped output:
    python -B benchmarks/benchmark_publication.py PATH_TO_OUTPUT --repeats 3

This performs complete integrity verification; no files are written. Intended
for separate baseline/post-change runs on identical input and hardware.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from net_hdbscan.publication import _verify_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    opts = parser.parse_args()
    if opts.repeats < 1:
        parser.error('--repeats must be at least 1')
    dest = opts.output_dir.resolve(strict=True)
    if not dest.is_dir():
        parser.error('output_dir must be a directory')
    manifest = json.loads((dest / 'manifest.json').read_text(encoding='utf-8'))
    results = []
    for _ in range(opts.repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            info = _verify_generation(dest)
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        elapsed = time.perf_counter() - t0
        results.append({'seconds': elapsed, 'peak_python_tracemalloc_bytes': peak})
    print(json.dumps({
        'output_dir': str(dest),
        'run_type': manifest['run'],
        'n_artifacts': len(info),
        'artifact_bytes': sum((dest / rel).stat().st_size for rel in info),
        'repeats': results,
        'median_seconds': statistics.median(x['seconds'] for x in results),
        'note': 'tracemalloc does not include all native NumPy/Arrow allocations',
    }, indent=2))


if __name__ == '__main__':
    main()
