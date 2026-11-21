# Development, validation and performance verification

This guide contains maintainer-facing details deliberately kept out of the introductory [README](../README.md). The implementation is deterministic under its documented input conventions, but its scientific validity depends on the network-distance model selected by the user.

## Install and test locally

For the full CLI/file test environment:

```powershell
python -m pip install -e ".[dev]"
python -B -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Tests failed; commit/push/merge blocked" }
```

Use `python -m pip install -e ".[dev,osm]"` if you will also test live OSM retrieval. OSMnx is optional; the ordinary offline suite does not require contacting Overpass. Local failure-path tests should run in addition to success-path tests.

Current test coverage includes sparse graph validation, weighted and tie-heavy HDBSCAN*, equivalence with scikit-learn on applicable cases, order and ID invariance, snapping/topology, bounded shortest paths, neighbor-pair limits, adaptive-radius diagnostics, group category/sentinel handling, output manifests, OSM cache/selector failures, CLI errors, and publication rollback/recovery fault injection.

## CI requirements

The GitHub Actions workflow [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs the complete pytest suite on Ubuntu and Windows with Python 3.11 and 3.12. It triggers on pull requests targeting `main` and on pushes to `main` or `wip/**` (not automatically on pushes to every feature-branch name).

For pull requests, **every applicable job must be completed successfully** before merging. Missing, pending, canceled, timed-out, skipped or failed jobs block the merge. CI success on a previous commit does not authorize merging a new head SHA. Explicitly verify the current head, base, workflow results, and GitHub merge state immediately before the merge operation. Do not use direct main-branch pushes as a substitute for required review/CI gates.

After merging, separately check the `main` branch's push-triggered CI and (where applicable) documentation deployment. These cannot be assumed successful from the PR checks alone.

## Hierarchy extraction performance

`compare_extractions()` fits the hierarchy once and selects EOM and leaf variants against that same fitted object. Mapping selected EOM nodes to selected leaf descendants uses a single topological pass through the condensed hierarchy instead of a nested pairwise ancestry search. This changes the mapping's computational work, not the clustering's selected partition.

Avoid claiming a production speedup solely from a synthetic microbenchmark. Report graph size, hierarchy size, selected-cluster counts, Python/dependency versions, machine, and repeated wall times when comparing versions.

## Output integrity and publication benchmark

Within one complete integrity scan, each output artifact is decoded once and reused for partition/cardinality verification; SHA-256 hashing and cross-file integrity checks remain mandatory. **A publication operation deliberately verifies the generation at more than one phase**, because the on-disk state changes across staging, renaming, and recovery. The number of integrity scans is not the same as the number of artifact reads inside one scan.

Run the read-only benchmark against an existing representative output generation:

```powershell
python -B benchmarks/benchmark_publication.py "PATH_TO_GROUPED_OUTPUT" --repeats 3
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed" }
```

Compare the same input generation and artifact sizes on the same machine, storage, Python version, and dependencies. Record every run and the median. The benchmark reports total artifact bytes, scan times, and **Python-traced** peak allocations. NumPy, Arrow, other native allocations, filesystem cache effects, and process peak RSS are not fully represented by `tracemalloc`. Do not treat the result as a whole-process memory measurement or imply a speedup without a controlled comparison.

Large grouped outputs remain a required integration/performance test before claiming production scalability. Unit tests and microbenchmarks alone do not establish real-world throughput.

## Publication failure and recovery semantics

File outputs are staged, checked, and then published using a two-phase backup/journal protocol. The operation is **not** a globally atomic multi-file transaction on all platforms. During a rename interval or after a crash, readers may temporarily find no destination until journal recovery.

A journal in the `prepared` phase permits rollback only to an independently verified complete predecessor. After an output is committed, predecessor backup deletion and journal removal are housekeeping. If they fail, `PostCommitCleanupError` stops execution while retaining the verified new generation and the recovery journal. **Never roll back to a backup that may have been partially removed.** An unreadable journal or missing/damaged committed output requires investigation rather than speculative automatic deletion.

Automated workflows must stop immediately on errors. A failed test, failed artifact verification, incomplete CI, or unexpected branch/SHA must block commits, pushes, merges, deployments, and destructive operations. Exercise injected failures as well as the happy path.

## OSM integration validation

The OSM adapter requires explicit bbox and exactly one network selector. Its cache records the retrieval time, source, OSMnx version, projection, SHA-256, and attribution. Network acquisition must be tested separately from clustering. For a live smoke test, use a deliberately small bounding box, record the resulting cache path/provenance, and verify a second identical request is a validated cache hit rather than another network query.

OSM's way filters are **not** an implementation of travel-time, one-way rules, river flow, or ecological dispersal. Do not infer that a successful data download validates the scientific distance model. See [Network sources](NETWORK_SOURCES.md).
