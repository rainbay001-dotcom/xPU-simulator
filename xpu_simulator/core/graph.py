"""Computation graph representation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx

from .operator import OpSpec, TensorSpec


@dataclass
class Node:
    """A node in the computation graph representing an operation."""
    id: int
    op: OpSpec
    name: Optional[str] = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        label = self.name or self.op.name or self.op.op_type.name
        return f"Node({self.id}, {label})"


class ComputeGraph:
    """Directed acyclic graph of operations, wrapping networkx.DiGraph."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self._graph = nx.DiGraph()
        self._next_id = 0

    def add_node(self, op: OpSpec, name: Optional[str] = None) -> Node:
        """Add an operation node to the graph."""
        node = Node(id=self._next_id, op=op, name=name)
        self._next_id += 1
        self._graph.add_node(node)
        return node

    def add_edge(self, src: Node, dst: Node, tensor: Optional[TensorSpec] = None):
        """Add a data dependency edge from src to dst."""
        self._graph.add_edge(src, dst, tensor=tensor)

    def topo_order(self) -> list[Node]:
        """Return nodes in topological order."""
        return list(nx.topological_sort(self._graph))

    def predecessors(self, node: Node) -> list[Node]:
        return list(self._graph.predecessors(node))

    def successors(self, node: Node) -> list[Node]:
        return list(self._graph.successors(node))

    @property
    def nodes(self) -> list[Node]:
        return list(self._graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Access underlying networkx graph for advanced analysis."""
        return self._graph

    def __repr__(self):
        return f"ComputeGraph({self.name!r}, nodes={self.num_nodes}, edges={self.num_edges})"
