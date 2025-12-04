"""
Utility functions for pyrivergraph.

This module provides shared utility functions used across the rivergraph package,
including geometric calculations, data structure helpers, and common algorithms.
"""

import math
import numpy as np
from typing import List, Tuple, Set, Dict, Any, Optional, Union
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)




def find_cycles_dfs(adjacency_dict: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find all cycles in a directed graph using depth-first search.

    Args:
        adjacency_dict: Dictionary mapping node_id -> list of connected node_ids

    Returns:
        List of cycles, where each cycle is a list of node IDs
    """
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node: int) -> None:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in adjacency_dict.get(node, []):
            dfs(neighbor)

        rec_stack.remove(node)
        path.pop()

    # Start DFS from each unvisited node
    for node in adjacency_dict:
        if node not in visited:
            dfs(node)

    return cycles


def topological_sort(adjacency_dict: Dict[int, List[int]]) -> List[int]:
    """
    Perform topological sort on a directed acyclic graph.

    Args:
        adjacency_dict: Dictionary mapping node_id -> list of connected node_ids

    Returns:
        Topologically sorted list of node IDs

    Raises:
        ValueError: If the graph contains cycles
    """
    # Calculate in-degrees
    in_degree = defaultdict(int)
    all_nodes = set(adjacency_dict.keys())

    for node in adjacency_dict:
        for neighbor in adjacency_dict[node]:
            in_degree[neighbor] += 1
            all_nodes.add(neighbor)

    # Initialize queue with nodes having no incoming edges
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Remove this node from graph and update in-degrees
        for neighbor in adjacency_dict.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all nodes were processed (no cycles)
    if len(result) != len(all_nodes):
        raise ValueError("Graph contains cycles - topological sort not possible")

    return result


def find_strongly_connected_components(adjacency_dict: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find strongly connected components using Tarjan's algorithm.

    Args:
        adjacency_dict: Dictionary mapping node_id -> list of connected node_ids

    Returns:
        List of strongly connected components, each as a list of node IDs
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    components = []

    def strongconnect(node: int) -> None:
        # Set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        # Consider successors of node
        for neighbor in adjacency_dict.get(node, []):
            if neighbor not in index:
                # Successor has not yet been visited; recurse on it
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack.get(neighbor, False):
                # Successor is in stack and hence in the current SCC
                lowlinks[node] = min(lowlinks[node], index[neighbor])

        # If node is a root node, pop the stack and create an SCC
        if lowlinks[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            components.append(component)

    # Run the algorithm for each unvisited node
    for node in adjacency_dict:
        if node not in index:
            strongconnect(node)

    return components



