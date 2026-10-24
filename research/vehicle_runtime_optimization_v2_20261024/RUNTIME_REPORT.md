# Runtime adapter validation

No vehicle experiment was repeated and no frozen policy was modified. Saved packet inputs were replayed. This separate adapter uses SQLite WAL with FULL synchronization and a checked single-owner connection reused by the original authentication code; the original per-packet authentication transactions and commits remain. Frozen forest predictions use the same float32 splits and float64 tree accumulation, with the native implementation retained for unsupported/nonfinite inputs. No learned weights or decision thresholds changed.

## Matched sequential timing

12 saved scenarios (all12conditions,3profiles assigned cyclically, predeclared seed2467),200queries each, identical inputs. Original/pooled-native/pooled-forest implementations rotate order across scenarios. Every measured sample and outlier is retained. This is engineering replay, not a new independent research confirmation. The old three-worker wall times are not the paired baseline.

| Implementation | Queries | p50 ms | p95 ms | p99 ms | Maximum ms | Loops >100ms |
|---|---:|---:|---:|---:|---:|---:|
| original | 2400 | 19.034 | 22.688 | 29.857 | 403.734 | 16 |
| pool_native | 2400 | 7.785 | 9.134 | 11.081 | 174.683 | 2 |
| pool_fast | 2400 | 6.057 | 7.415 | 9.579 | 171.847 | 2 |

Per-implementation initialization and checkpoint-close times are retained separately in runtime_summary.json. Fixed-period replay time excludes file loading, post-measurement output comparisons, plotting, initialization and closing. It includes policy processing, command/certificate construction and actuation checks. Warm caches, system scheduling and shared Windows activity limit transferability. Descriptive intervals resample12 matched scenario blocks at one seed; they do not establish a population deadline bound.

## Behavior and persistence

All216joint-policy saved trials,43200decisions across6seeds,3profiles and12conditions, match the original policy outputs (outside nested upstream diagnostics), issued signed command/certificate bytes, accepted commands/statuses and operator cards exactly. Nested upstream diagnostic floats are compared at1e-12; this tolerance is not applied to issued packets, policy outputs or cards. Full equivalence reuses12completed combined-adapter benchmark checkpoints and checks the remaining204 in3isolated worker processes in fresh state. No simulator, new trajectories, physical measurements or revised collision counts are generated.

Seventeen focused tests cover frozen-tree predictions including observed features and split neighbors, native fallback, FULL/WAL settings, saved command equivalence, forged packets, watchdog expiry, owner-thread enforcement, replay-state reopening and committed/uncommitted recovery after abrupt exits of dedicated test subprocesses. This is process-exit recovery, not a power-loss test or complete controller-state recovery. Restart requires the original qualified-session procedure; this adapter does not invent bootstrap trust.

## SQLite and execution scope

The device SQLite version is3.49.1. SQLite documents a WAL-reset race in older versions when concurrent connections write/checkpoint from separate threads/processes. This adapter requires one owner thread/process and sequential database writes; its cached connection keeps every original FULL commit, revalidates database file identity on every loan, and checkpoints after all loans are released. Concurrent external writers/checkpointers and network filesystems are outside this runtime qualification. This does not certify concurrent operation on the installed SQLite version. https://sqlite.org/wal.html . FULL synchronization is retained, not replaced by NORMAL/OFF; physical durability remains dependent on the storage system. https://sqlite.org/pragma.html#pragma_synchronous .

A finite replay benchmark cannot establish hard real-time scheduling or latency-aware closed-loop safety. The original simulation's exact-time delivery/cadence premises remain. The adapter is opt-in, single-owner and qualified for the joint method only. No manuscript or journal-acceptance claim is produced.
