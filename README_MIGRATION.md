# Removing `rivergraph` from pyflowline in favor of pyearthriver as a dependency

**Status**: Draft for discussion
**Date**: 2026-08-27
**Scope**: Evaluate replacing `pyflowline.classes.rivergraph.pyrivergraph` with the
standalone `pyearthriver` package.

---

## 1. Background

`pyflowline/classes/rivergraph.py` (the `pyrivergraph` class, ~2,300 lines of
graph-based river-network algorithms) has been extracted into the standalone
package **pyearthriver** so it can be reused by other projects beyond pyflowline.

The goal of this document is to decide whether pyflowline should **delete its own
copy** and consume `pyearthriver` as a normal dependency instead.

### 1.1 Current state of the two copies

| Item | pyflowline copy | pyearthriver copy | Identical? |
|---|---|---|---|
| `classes/rivergraph.py` | `pyrivergraph` | `pyrivergraph` | Yes, except 8 import lines |
| `classes/flowline.py` | `pyflowline` class | `pyflowline` class | Functionally yes (see note) |
| `classes/vertex.py` | `pyvertex` | `pyvertex` | Yes (formatting only) |
| `classes/edge.py` | `pyedge` | `pyedge` | Yes (formatting only) |
| `classes/confluence.py` | `pyconfluence` | `pyconfluence` | Yes (formatting only) |
| `algorithms/auxiliary/find_*` | duplicated | duplicated | Yes |
| `algorithms/cython/kernel.pyx` | duplicated | duplicated | Nearly identical |
| `formats/export_flowline.py`, `export_vertex.py` | richer (extra export options) | simplified | **No** — pyflowline keeps its own |

> **Verified by line-level diff (2026-08-27)**:
> - `pyvertex` / `pyedge` / `pyconfluence` differ only in formatting (quote
>   style, line wrapping) — zero functional differences.
> - `pyflowline` differs by exactly **one extra method** in pyflowline:
>   `calculate_distance_to_vertex`, which is **dead code** (defined but never
>   called anywhere in pyflowline). It can be dropped or trivially added to
>   pyearthriver for completeness.
> - The four core classes therefore **already live in pyearthriver** and can
>   serve as the single source of truth as-is.
> - The exporters are the only real divergence, but this is acceptable:
>   pyflowline's pipeline calls its own exporters directly; `pyrivergraph` uses
>   exporters only for internal debug dumps, so each package keeps its own.

### 1.2 How pyflowline currently uses `pyrivergraph`

Only two files consume the class:

- `pyflowline/algorithms/process_flowline.py` — uses `split_flowline`,
  `remove_duplicate_flowlines`, `remove_disconnected_flowlines`, `merge_flowline`,
  `remove_braided_river`, `remove_parallel_river`, `remove_cycle`,
  `update_headwater_stream_order`, `define_river_confluence`, `define_stream_order`,
  `remove_small_river`, `define_stream_segment`, `define_stream_topology`,
  `define_stream_order`.
- `pyflowline/classes/basin.py` — uses `find_confluence`, `build_confluence`,
  `split_flowline`.

### 1.3 The critical coupling point

`pyrivergraph` is **not a self-contained module**: it both consumes and produces
`pyvertex` / `pyflowline` / `pyedge` / `pyconfluence` objects. Those classes are
used throughout pyflowline (mesh generation modules, formats I/O, basin processing).

Facts that shape the decision:

- `pyrivergraph` contains **no `isinstance` checks** on its inputs, so pyflowline
  objects can be passed in (duck typing works today).
- However, the methods **return newly constructed objects** built from the classes
  imported inside `rivergraph.py`. If pyflowline delegates to pyearthriver, all
  flowlines/vertices/confluences coming *out* of the graph become
  `pyearthriver.classes.*` instances while pyflowline's own modules still expect
  `pyflowline.classes.*` instances.
- Any downstream code that does `isinstance(...)`, pickling across class
  identities, or relies on pyflowline-only attributes/methods will break.

**Consequence**: "remove only `rivergraph.py`" is not a drop-in change. The class
boundary (vertex/edge/flowline/confluence + export helpers) must move together with
the graph, or pyflowline must rebase its own classes onto pyearthriver's.

---

## 2. Pros of switching to pyearthriver as a dependency

1. **Single source of truth.** One implementation of the graph algorithms instead
   of two. Bug fixes and enhancements land once and benefit all consumers.
2. **The divergence problem is already real.** The four supporting classes already
   differ between repos; keeping two copies guarantees silent behavioral drift.
3. **Reuse in other projects.** pyearthriver is designed as a general-purpose river
   network analysis library; other tools can depend on it without pulling in the
   whole pyflowline mesh-generation stack.
4. **Smaller, more focused pyflowline.** pyflowline shrinks back to its core
   mission (mesh-independent river network generation for hydrologic models);
   ~2,300 lines of algorithm code plus duplicated helpers leave the repo.
5. **Independent release cadence.** Algorithm improvements can be released as
   pyearthriver patch versions without a full pyflowline release, and vice versa.
6. **Cleaner architecture.** Forces an explicit, documented public API between the
   graph engine and its consumers, instead of implicit in-repo coupling.
7. **Consolidated testing/CI.** Graph algorithm tests live with the graph code and
   run in one CI pipeline instead of being duplicated.
8. **The planned modular refactor fits better upstream.** pyflowline's
   `pyrivergraph_modular_architecture.md` (splitting the monolithic file into ~10
   modules) should now be executed in pyearthriver, not in pyflowline — otherwise
   the two copies diverge even faster.

## 3. Cons and risks

1. **Cross-package object coupling (the main risk).** As described in §1.3,
   pyrivergraph returns objects of the classes it imports. Removing the copy means
   pyflowline either:
   - adopts pyearthriver's classes everywhere (larger refactor), or
   - keeps its own classes and accepts mixed object types (fragile).
   A half-way removal will produce subtle, hard-to-debug failures.
2. **API stability becomes a hard requirement.** Today pyflowline can change
   rivergraph internals freely. After the split, every pyrivergraph change must
   respect semantic versioning and pyflowline's pinned version range; breaking
   changes require coordinated releases.
3. ~~Diverged class features must be reconciled first.~~ **Resolved by diff
   verification**: the classes differ only in formatting, plus one unused
   method (`calculate_distance_to_vertex`). No reconciliation work is needed;
   pyearthriver's classes can be adopted directly.
4. **Release and packaging coordination.** pyflowline must declare
   `pyearthriver>=X.Y` in its dependencies, and pyearthriver must be published to
   PyPI *before* any pyflowline release that needs it. Local editable installs
   (`pip install -e .` in both repos) are needed during development.
5. **Additional dependency surface.** pyearthriver pulls in `scipy` and `pyearth`.
   pyflowline already uses `pyearth` at runtime (but notably does **not declare
   it** in `pyproject.toml` — worth fixing regardless), so the net-new dependency
   is mainly `scipy`.
6. **Version-resolution surprises.** If other downstream projects pin pyflowline
   and pyearthriver separately, pip's resolver must keep them compatible; a
   careful pinning strategy (e.g. `pyearthriver>=0.2,<0.3`) is needed.
7. **Debugging across package boundaries.** Stack traces and `pip` provenance now
   span two packages; bug reports must capture both versions.
8. **Cython build duplication remains but moves upstream.** The `kernel.pyx`
   extension and its pre-compiled fallback must be maintained (and tested on all
   target platforms) in pyearthriver's CI instead of pyflowline's.
9. **Licensing.** pyearthriver is BSD-3-Clause while pyflowline declares a
   "custom" license. Depending *on* BSD-3 code is unproblematic, but the license
   texts should be reviewed once before release.
10. **Documentation churn.** pyflowline docs/examples referencing
    `pyflowline.classes.rivergraph` must be updated or backward-compatible
    re-exports must be kept for a deprecation period.

## 4. Options

### Option A — Full class migration (recommended)

pyflowline depends on pyearthriver and **re-exports** the core classes:

```python
# pyflowline/classes/vertex.py (deprecation shim)
from pyearthriver.classes.vertex import pyvertex  # noqa: F401
```

- Delete `rivergraph.py` from pyflowline; import `pyrivergraph` from pyearthriver.
- Merge any pyflowline-only class features into pyearthriver first.
- Keep re-export shims for one or two minor versions, then remove.
- **Pros**: zero mixed-type risk; true single source of truth.
- **Cons**: largest upfront effort; requires reconciling class differences.

### Option B — Remove only rivergraph.py, keep pyflowline classes

- pyflowline imports `pyrivergraph` from pyearthriver but keeps its own classes.
- **Pros**: smallest diff.
- **Cons**: mixed object types (pyearthriver outputs vs. pyflowline internals);
  high risk of silent breakage; not recommended unless rivergraph is refactored to
  be factory-injected (see Option C).

### Option C — Dependency-injected / protocol-based graph engine

- Redesign `pyrivergraph` to accept class factories (or structural Protocols) for
  vertex/flowline/edge/confluence so it constructs caller-side types.
- **Pros**: clean long-term API; truly reusable engine.
- **Cons**: significant redesign of a 2,300-line class before any benefit.

### Option D — Keep two copies (status quo)

- **Pros**: no migration work; no cross-repo release coupling.
- **Cons**: guarantees continued drift (already observed), duplicate bug fixes,
  duplicated tests/CI — the exact problem motivating this proposal.

## 5. Recommendation

Adopt **Option A** — decided 2026-08-27 to skip the deprecation shims and
switch pyflowline to **direct pyearthriver imports** (hard break for anyone
importing the removed `pyflowline.classes.{vertex,edge,flowline,confluence,rivergraph}`
modules; acceptable since pyflowline ships a new minor version anyway).

Status of each step:

1. ~~Reconcile the diverged classes~~ — **verified unnecessary**; the one
   functional difference (`calculate_distance_to_vertex`) was ported into
   pyearthriver. ✅ done
2. **Publish** pyearthriver `0.2.0` to PyPI and verify `pip install
   pyearthriver` (Cython extension + pure-Python fallback) on all platforms.
   ⏳ pending
3. **Refactor pyflowline** — done directly (no shims): ✅ done
   - deleted `classes/{vertex,edge,flowline,confluence,rivergraph}.py`
   - rewrote all 19 importing modules to import from `pyearthriver.classes.*`
   - added `pyearth` and `pyearthriver>=0.2.0` to `pyproject.toml`
4. **Run** the full pyflowline test/example suite (Susquehanna examples) and
   diff output geojson against a pre-migration run. ⏳ pending (requires a
   Python environment with gdal/netCDF4)
5. ~~Deprecation shims~~ — skipped by decision. ❌ n/a
6. **Move** the modular-architecture plan (`pyrivergraph_modular_architecture.md`)
   into the pyearthriver repo and execute it there. ⏳ pending

## 6. Preconditions / checklist

- [x] Diff-level merge of the four core classes (verified 2026-08-27: only
      formatting differences + one unused method)
- [x] `calculate_distance_to_vertex` ported into pyearthriver
- [x] pyflowline refactored to import directly from pyearthriver (no shims);
      five duplicate modules deleted; `pyearthriver>=0.2.0` declared
- [ ] pyearthriver CI builds the Cython kernel on Linux/macOS/Windows
- [ ] pyearthriver public API frozen and documented (`pyrivergraph` method surface
      used by pyflowline listed in §1.2)
- [ ] Version pinning policy defined (e.g. `pyearthriver>=0.2,<0.3`)
- [ ] pyflowline `pyproject.toml`: add `pyearthriver` (and the currently
      undeclared `pyearth`) to dependencies
- [ ] Regression test: identical geojson outputs before/after migration
- [ ] License review (custom vs. BSD-3-Clause)
- [ ] Update pyflowline README/docs and examples

---

*This document lives in the pyearthriver repository; the final decision and the
code changes primarily affect pyflowline.*
