"""
Cycle detection and resolution algorithms using reverse-BFS from the outlet.
"""

import logging
from collections import deque
from typing import List, Set, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyearthriver.classes.rivergraph import pyrivergraph
    from pyearthriver.classes.flowline import pyflowline

logger = logging.getLogger(__name__)


def remove_cycles_from_graph(graph: "pyrivergraph") -> List["pyflowline"]:
    """
    Remove directed cycles from a river network graph using reverse-BFS from the outlet.

    Algorithm:
    A valid flow network is a DAG where every flowline points toward the
    outlet. This algorithm grows a set `known` of vertices confirmed to have
    a path to the outlet, starting from the outlet itself and expanding upstream.

    For each flowline start -> end:
    - If end in known and start not in known:
        Flowline correctly flows toward outlet. Add start to known and expand.
    - If end in known and start in known:
        Flowline creates a redundant/cyclic loop. Remove this flowline.
    - If end not in known:
        Skip for now until end is discovered.

    Args:
        graph (pyrivergraph): The river graph instance.

    Returns:
        List[pyflowline]: Acyclic list of retained flowlines.
    """
    if not hasattr(graph, "aFlowline") or not graph.aFlowline:
        logger.warning("No flowlines available in class instance for cycle removal")
        return []

    if len(graph.aFlowline) <= 1:
        logger.debug("Skipping cycle removal: insufficient flowlines")
        return graph.aFlowline.copy()

    logger.info(
        f"Removing cycles from {len(graph.aFlowline)} flowlines using reverse-BFS approach"
    )

    outlet_id = getattr(graph, "pVertex_outlet_id", None)

    if outlet_id is None:
        logger.warning(
            "remove_cycle: no outlet vertex available — "
            "cycle removal skipped (outlet is required to determine flow direction)"
        )
        return graph.aFlowline.copy()

    # Build reverse adjacency: end_id -> [(start_id, flowline_idx), ...]
    reverse_adj: Dict[int, List[Tuple[int, int]]] = {}
    for fl_idx, (s_id, e_id) in graph.aFlowline_edges.items():
        reverse_adj.setdefault(e_id, []).append((s_id, fl_idx))

    known_vertices: Set[int] = {outlet_id}
    flowlines_to_remove: Set[int] = set()

    queue = deque([outlet_id])

    while queue:
        current_end = queue.popleft()

        for start_id, fl_idx in reverse_adj.get(current_end, []):
            if fl_idx >= len(graph.aFlowline):
                continue

            fl = graph.aFlowline[fl_idx]

            if start_id not in known_vertices:
                known_vertices.add(start_id)
                queue.append(start_id)
                logger.debug(
                    f"  [remove_cycle] Accepted flowline idx={fl_idx} "
                    f"({start_id} → {current_end})"
                )
            else:
                flowlines_to_remove.add(fl_idx)
                print(
                    f"  [remove_cycle] Cycle detected — removing flowline idx={fl_idx} "
                    f"start=({fl.pVertex_start.dLongitude_degree:.4f},"
                    f"{fl.pVertex_start.dLatitude_degree:.4f}) "
                    f"end=({fl.pVertex_end.dLongitude_degree:.4f},"
                    f"{fl.pVertex_end.dLatitude_degree:.4f}) "
                    f"order={fl.iStream_order} "
                    f"length={getattr(fl,'dLength',0):.1f}"
                )

    print(
        f"  [remove_cycle] Reverse-BFS complete: "
        f"known={len(known_vertices)} vertices, "
        f"removing {len(flowlines_to_remove)} cyclic flowline(s)"
    )
    print(f"  [remove_cycle] Total flowlines before removal: {len(graph.aFlowline)}")

    try:
        result = [
            graph.aFlowline[i]
            for i in range(len(graph.aFlowline))
            if i not in flowlines_to_remove
        ]
        print(f"  [remove_cycle] Total flowlines after removal: {len(result)}")

        if len(result) == 0 and len(graph.aFlowline) > 0:
            print("  [remove_cycle] WARNING: all flowlines removed, returning original")
            return graph.aFlowline.copy()

        graph._update_graph_flowlines(result)
        return result
    except Exception as e:
        logger.error(f"Error filtering flowlines during cycle removal: {e}")
        return graph.aFlowline.copy()
