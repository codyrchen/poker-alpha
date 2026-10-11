# Strategy artifact engineering (Phase II, Programs 8 and 17)

Release artifact: `holdem_v2_native_seed0_2m.npz`, 144,025 infosets and
522,336 action entries. It was **not modified**. Artifact:
`results/phase2/artifact_compression.json`. Reproduce:
`python experiments/phase2_artifact_compression.py`.

| variant | npz bytes | max TV | visit-weighted TV | top-action flips |
|---|---|---|---|---|
| float32 (release) | 2,917,740 | 0 | 0 | 0 |
| float16 | 2,132,655 | 2.0e-4 | 2.2e-5 | 8 |
| **uint16 (normalized)** | **2,143,676** | **3.7e-5** | **7.1e-6** | **0** |
| uint8 | 1,665,659 | 9.5e-3 | 1.4e-3 | 136 |
| drop probs < 1e-3 (sparse) | – | 4.6e-3 | 2.1e-4 | 0 |

For scale, the TV between two training seeds of the same config is 0.144
visit-weighted. uint16 storage is 27 % smaller with error five orders of
magnitude below training noise and no changed top action. It is a GO
candidate for a future artifact format v3, with a fresh content hash and a
reader that accepts both. This sprint does not change the release.

18 % of action entries are below 1e-3 and 31 % below 1e-2. Sparse storage
would save little next to the compressed npz, so it is a NO-GO as a separate
format.

**Load cost.** `load_artifact` takes 0.16 s median and serves about 2.6M dict
lookups/s, but raises peak RSS by about 157 MB for a 3 MB file. The cost is
the dict-of-dicts representation. The numeric arrays alone are 3.8 MB
uncompressed. A memory-mapped layout is feasible only with raw `.npy` members
or a custom binary file, because `.npz` members are not mmap-able. Binary
search over a sorted key array gives about 450k lookups/s with no index
build. An array-backed lazy mapping is the obvious memory fix (see
`docs/phase2/native_systems.md` if implemented).

## Distillation spike (approximation only)

Visit-weighted prototype policies of the release itself:

| prototype key | prototypes | infoset compression | visit-weighted TV | top-action agreement (visit-weighted) |
|---|---|---|---|---|
| betting context only (card-blind) | 460 | 313x | 0.406 | 62 % |
| strength rung + betting | 2,143 | 67x | 0.288 | 72 % |
| **all card features except texture** | 26,344 | 5.5x | **0.117** | 89 % |

Merging keys that differ only in board texture moves the policy by 0.117
TV, which is **less than the 0.144 TV between two training seeds**. On this
evidence the texture dimension carries less policy signal than training
noise. That is a candidate abstraction simplification, not a proven one,
because a TV below the noise floor does not establish value equivalence. It
is recorded as a self-generated hypothesis in the research report.
