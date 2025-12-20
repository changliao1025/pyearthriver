# PyRivergraph Modularization Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the modularization plan outlined in [`rivergraph_modularization_plan.md`](rivergraph_modularization_plan.md).

## Current Status

✅ **Phase 1 Complete**: Directory structure and `__init__.py` files created

### Created Structure

```
rivergraph/
├── __init__.py                    ✅ Created
├── classes/
│   ├── __init__.py                ✅ Created
│   ├── vertex.py                  ✅ Existing
│   ├── flowline.py                ✅ Existing
│   ├── confluence.py              ✅ Existing
│   ├── edge.py                    ✅ Existing
│   └── rivergraph.py              ✅ Existing (to be refactored)
├── core/
│   └── __init__.py                ✅ Created
├── operations/
│   └── __init__.py                ✅ Created
├── analysis/
│   └── __init__.py                ✅ Created
└── formats/
    └── __init__.py                ✅ Created
```

## Next Steps

### Phase 2: Extract Core Graph Module

This is a large refactoring task. Given the complexity, I recommend the following approach:

#### Option A: Incremental Refactoring (Recommended)

**Advantages:**
- Lower risk of breaking existing functionality
- Can test after each small change
- Easier to review and debug
- Can be done over multiple sessions

**Steps:**
1. Keep the existing [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1) working
2. Create new modules alongside it
3. Gradually move functionality
4. Update imports incrementally
5. Remove old code only after new code is tested

#### Option B: Complete Rewrite

**Advantages:**
- Clean slate, better architecture from start
- No legacy code to maintain during transition

**Disadvantages:**
- Higher risk
- Requires comprehensive testing before deployment
- All-or-nothing approach

## Recommended Implementation Strategy

### Step 1: Create Core Graph Module (Estimated: 1-2 days)

Create [`rivergraph/core/graph.py`](../rivergraph/core/graph.py:1) with the `RiverGraph` class containing:

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 48-142: Initialization and basic graph operations
- Lines 1594-1679: Graph construction and management
- Keep it focused on data structure only

**Key Methods to Extract:**
```python
class RiverGraph:
    def __init__(self, flowlines, pVertex_outlet=None)
    def get_sources(self)
    def get_sinks(self)
    def get_vertices(self)
    def get_vertex_by_id(self, vertex_id)
    def get_vertex_id(self, vertex)
    def get_vertex_count(self)
    def _build_graph(self)
    def _set_outlet_vertex_id(self)
    def _update_graph_flowlines(self, new_flowlines)
```

### Step 2: Create Analysis Modules (Estimated: 2-3 days)

These are relatively independent and can be extracted first:

#### 2a. Create [`rivergraph/analysis/detection.py`](../rivergraph/analysis/detection.py:1)

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 738-1111: Detection methods

**Key Class:**
```python
class NetworkAnalyzer:
    def __init__(self, graph: RiverGraph)
    def find_braided_channels(self)
    def find_parallel_paths(self)
    def detect_cycles(self)
    # ... private helpers
```

#### 2b. Create [`rivergraph/analysis/pathfinding.py`](../rivergraph/analysis/pathfinding.py:1)

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 1684-1803: Path finding methods
- Lines 1930-1969: Upstream/downstream helpers

**Key Class:**
```python
class PathFinder:
    def __init__(self, graph: RiverGraph)
    def find_all_paths(self, start_id, target_id, max_depth=10)
    def path_to_flowlines(self, path)
    def find_outlet_reachable_vertices(self, outlet_vertex_id)
    def get_upstream_indices(self, flowline)
    def get_downstream_indices(self, flowline)
```

### Step 3: Create Operations Modules (Estimated: 3-4 days)

#### 3a. Create [`rivergraph/operations/simplification.py`](../rivergraph/operations/simplification.py:1)

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 147-733: Simplification methods

**Key Class:**
```python
class NetworkSimplifier:
    def __init__(self, graph: RiverGraph, analyzer: NetworkAnalyzer, pathfinder: PathFinder)
    def remove_disconnected_flowlines(self, outlet_vertex=None)
    def remove_braided_river(self)
    def remove_parallel_river(self)
    def remove_cycle(self)
    def remove_small_river(self, threshold, iterations=3, ...)
    def remove_duplicate_flowlines(self, direction_insensitive=False)
```

#### 3b. Create [`rivergraph/operations/modification.py`](../rivergraph/operations/modification.py:1)

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 1116-1308: Modification methods

**Key Class:**
```python
class NetworkModifier:
    def __init__(self, graph: RiverGraph, analyzer: NetworkAnalyzer)
    def split_flowline(self, vertices=None, flag_intersect=None, flag_use_id=None)
    def merge_flowline(self)
```

#### 3c. Create [`rivergraph/operations/topology.py`](../rivergraph/operations/topology.py:1)

**Extract from [`rivergraph.py`](../rivergraph/classes/rivergraph.py:1):**
- Lines 1314-1590: Topology methods
- Lines 1836-1874: Confluence management
- Lines 1879-1929: Stream order processing

**Key Class:**
```python
class TopologyManager:
    def __init__(self, graph: RiverGraph, pathfinder: PathFinder)
    def define_stream_segment(self)
    def define_river_confluence(self)
    def define_stream_topology(self)
    def define_stream_order(self, method=1)
    def update_headwater_stream_order(self)
    def identify_headwater_flowlines(self)
```

### Step 4: Create Facade Class (Estimated: 1 day)

Create [`rivergraph/core/rivergraph.py`](../rivergraph/core/rivergraph.py:1) as a thin wrapper:

```python
from rivergraph.core.graph import RiverGraph
from rivergraph.operations.simplification import NetworkSimplifier
from rivergraph.operations.modification import NetworkModifier
from rivergraph.operations.topology import TopologyManager
from rivergraph.analysis.detection import NetworkAnalyzer
from rivergraph.analysis.pathfinding import PathFinder

class pyrivergraph:
    """
    Main facade class for river network analysis.

    Maintains backward compatibility while delegating to specialized modules.
    """

    def __init__(self, flowlines, pVertex_outlet=None):
        # Initialize core graph
        self._graph = RiverGraph(flowlines, pVertex_outlet)

        # Initialize analysis components
        self._analyzer = NetworkAnalyzer(self._graph)
        self._pathfinder = PathFinder(self._graph)

        # Initialize operation components
        self._simplifier = NetworkSimplifier(self._graph, self._analyzer, self._pathfinder)
        self._modifier = NetworkModifier(self._graph, self._analyzer)
        self._topology = TopologyManager(self._graph, self._pathfinder)

        # Expose graph properties for backward compatibility
        self.aFlowline = self._graph.aFlowline
        self.aVertex = self._graph.aVertex
        self.pVertex_outlet = self._graph.pVertex_outlet
        # ... other properties

    # Delegate methods to appropriate modules
    def remove_braided_river(self):
        result = self._simplifier.remove_braided_river()
        self._sync_state()
        return result

    def detect_cycles(self):
        return self._analyzer.detect_cycles()

    def merge_flowline(self):
        result = self._modifier.merge_flowline()
        self._sync_state()
        return result

    def define_stream_order(self, method=1):
        result = self._topology.define_stream_order(method)
        self._sync_state()
        return result

    def _sync_state(self):
        """Sync state after operations that modify the graph"""
        self.aFlowline = self._graph.aFlowline
        self.aVertex = self._graph.aVertex
        # ... sync other properties

    # ... delegate all other public methods
```

### Step 5: Update Package Imports (Estimated: 0.5 days)

Update [`rivergraph/__init__.py`](../rivergraph/__init__.py:1):

```python
# Import from new location
from rivergraph.core.rivergraph import pyrivergraph

# Keep backward compatibility
from rivergraph.classes.vertex import pyvertex
from rivergraph.classes.flowline import pyflowline
from rivergraph.classes.edge import pyedge
from rivergraph.classes.confluence import pyconfluence

__all__ = [
    'pyrivergraph',
    'pyvertex',
    'pyflowline',
    'pyedge',
    'pyconfluence',
]
```

Update [`rivergraph/classes/__init__.py`](../rivergraph/classes/__init__.py:1):

```python
# Backward compatibility - import from new location
from rivergraph.core.rivergraph import pyrivergraph

# Keep existing imports
from .vertex import pyvertex
from .flowline import pyflowline
from .edge import pyedge
from .confluence import pyconfluence

__all__ = [
    'pyvertex',
    'pyflowline',
    'pyedge',
    'pyconfluence',
    'pyrivergraph',  # For backward compatibility
]
```

### Step 6: Testing Strategy

After each phase:

1. **Unit Tests**: Test new modules independently
2. **Integration Tests**: Test module interactions
3. **Regression Tests**: Run existing tests against new code
4. **Performance Tests**: Ensure no significant slowdown

## Practical Considerations

### Managing the Transition

1. **Keep Old Code**: Don't delete [`rivergraph/classes/rivergraph.py`](../rivergraph/classes/rivergraph.py:1) until new code is fully tested
2. **Feature Flag**: Consider using a feature flag to switch between old and new implementations
3. **Gradual Migration**: Users can gradually migrate to new imports

### Handling Dependencies

The modules have dependencies on each other:

```
RiverGraph (core)
    ↓
PathFinder, NetworkAnalyzer (analysis)
    ↓
NetworkSimplifier, NetworkModifier, TopologyManager (operations)
    ↓
pyrivergraph (facade)
```

Extract in this order to minimize circular dependencies.

### Code Duplication

Some helper methods are used by multiple modules. Options:

1. **Keep in original module**: If primarily used by one module
2. **Create utils module**: For truly shared utilities
3. **Duplicate**: If the code is simple and coupling is undesirable

## Estimated Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Directory structure | ✅ Complete |
| 2 | Core graph module | 1-2 days |
| 3a | Analysis: detection | 1 day |
| 3b | Analysis: pathfinding | 1 day |
| 4a | Operations: simplification | 1.5 days |
| 4b | Operations: modification | 0.5 days |
| 4c | Operations: topology | 1 day |
| 5 | Facade class | 1 day |
| 6 | Update imports | 0.5 days |
| 7 | Testing & debugging | 2-3 days |
| 8 | Documentation | 1 day |

**Total**: 10-13 days (2-3 weeks)

## Success Criteria

- ✅ All existing tests pass
- ✅ No breaking changes to public API
- ✅ Each module <500 lines
- ✅ Code coverage maintained or improved
- ✅ Performance within 5% of original
- ✅ Documentation updated

## Next Actions

Given the scope of this refactoring, I recommend:

1. **Review the plan** with stakeholders
2. **Create comprehensive tests** for existing functionality first
3. **Start with Phase 2** (Core Graph) as a proof of concept
4. **Get feedback** before proceeding with remaining phases
5. **Consider doing this incrementally** over multiple PRs/commits

Would you like me to:
- A) Start implementing Phase 2 (Core Graph module)?
- B) Create a test suite first?
- C) Create a detailed code extraction script?
- D) Something else?