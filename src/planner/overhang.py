"""
Overhang detection for FBS walls.

Determines max overhang based on edge type:
- Exterior contour → 0mm
- Interior joint → 500mm
- Door opening → 200mm
- Monolith → always 0mm
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

Point = Tuple[float, float]


class EdgeType(Enum):
    FREE_EDGE = 0
    DOOR_OPENING = 1
    EXTERIOR_JOINT = 2
    INTERIOR_JOINT = 3


OVERHANG_LIMITS = {
    EdgeType.FREE_EDGE: 0,
    EdgeType.DOOR_OPENING: 200,
    EdgeType.EXTERIOR_JOINT: 0,
    EdgeType.INTERIOR_JOINT: 500,
}


@dataclass
class EdgeConstraint:
    edge_type: EdgeType
    max_overhang_mm: int
    inward_direction: Optional[Tuple[float, float]] = None
    neighbor_node: Optional[Point] = None


@dataclass
class WallOverhangConstraints:
    wall_id: str
    left_edge: EdgeConstraint
    right_edge: EdgeConstraint


def normalize_point(p: Point, precision: int = 6) -> Point:
    """Normalize point to avoid float comparison issues."""
    return (round(p[0], precision), round(p[1], precision))


def point_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def cross_product_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """2D cross product."""
    return v1[0] * v2[1] - v1[1] * v2[0]


class OverhangAnalyzer:
    """
    Analyzes wall graph to determine overhang constraints.

    1. Find exterior contour via clockwise traversal
    2. Identify door openings (degree-1 nodes 800-1100mm apart)
    3. Classify each wall edge
    """

    DOOR_MIN_MM = 800
    DOOR_MAX_MM = 1100

    def __init__(self, G: nx.Graph):
        self.G = G
        self.exterior_edges: Set[Tuple[Point, Point]] = set()
        self.exterior_inward_normals: Dict[
            Tuple[Point, Point], Tuple[float, float]
        ] = {}
        self.door_openings: List[Tuple[Point, Point]] = []
        self.door_nodes: Set[Point] = set()

        self._find_exterior_contour()
        self._find_door_openings()

    def _find_leftmost_bottom_node(self) -> Optional[Point]:
        """Find leftmost-bottom node (guaranteed on exterior)."""
        if not self.G.nodes():
            return None
        return min(self.G.nodes(), key=lambda n: (n[0], n[1]))

    def _get_outgoing_angle(self, current: Point, next_node: Point) -> float:
        """Get angle of edge (radians, 0 = right)."""
        dx = next_node[0] - current[0]
        dy = next_node[1] - current[1]
        return np.arctan2(dy, dx)

    def _find_rightmost_turn(
        self, current: Point, prev: Optional[Point]
    ) -> Optional[Point]:
        """Find neighbor for clockwise traversal."""
        neighbors = [n for n in self.G.neighbors(current) if n != prev]
        if not neighbors:
            return None
        if len(neighbors) == 1:
            return neighbors[0]

        incoming_angle = self._get_outgoing_angle(prev, current) if prev else np.pi

        def turn_angle(neighbor):
            outgoing = self._get_outgoing_angle(current, neighbor)
            diff = outgoing - incoming_angle
            while diff < 0:
                diff += 2 * np.pi
            while diff >= 2 * np.pi:
                diff -= 2 * np.pi
            return diff

        return min(neighbors, key=turn_angle)

    def _find_exterior_contour(self):
        """Find exterior by clockwise traversal. Interior always on RIGHT."""
        start = self._find_leftmost_bottom_node()
        if start is None:
            return

        visited_edges: Set[Tuple[Point, Point]] = set()
        current = start
        prev = None

        max_iterations = len(self.G.edges()) * 2 + 10
        for _ in range(max_iterations):
            next_node = self._find_rightmost_turn(current, prev)
            if next_node is None:
                break

            edge_key = (current, next_node)
            if edge_key in visited_edges:
                break

            visited_edges.add(edge_key)
            self.exterior_edges.add(edge_key)

            dx = next_node[0] - current[0]
            dy = next_node[1] - current[1]
            length = (dx**2 + dy**2) ** 0.5
            if length > 1e-9:
                inward_normal = (dy / length, -dx / length)
                self.exterior_inward_normals[edge_key] = inward_normal

            prev = current
            current = next_node

            if current == start and prev is not None:
                break

    def _find_door_openings(self):
        """Find door openings: degree-1 nodes 800-1100mm apart."""
        degree_1_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 1]

        for i, n1 in enumerate(degree_1_nodes):
            for n2 in degree_1_nodes[i + 1 :]:
                dist = point_distance(n1, n2)
                if self.DOOR_MIN_MM <= dist <= self.DOOR_MAX_MM:
                    self.door_openings.append((n1, n2))
                    self.door_nodes.add(n1)
                    self.door_nodes.add(n2)

    def _is_edge_on_exterior(self, u: Point, v: Point) -> bool:
        """Check if edge on exterior contour."""
        return (u, v) in self.exterior_edges or (v, u) in self.exterior_edges

    def _get_inward_direction_at_node(
        self, node: Point, edge_u: Point, edge_v: Point
    ) -> Optional[Tuple[float, float]]:
        """Get inward direction at node."""
        if (edge_u, edge_v) in self.exterior_inward_normals:
            return self.exterior_inward_normals[(edge_u, edge_v)]
        if (edge_v, edge_u) in self.exterior_inward_normals:
            normal = self.exterior_inward_normals[(edge_v, edge_u)]
            return (-normal[0], -normal[1])
        return None

    def _classify_node(
        self, node: Point, wall_edge_u: Point, wall_edge_v: Point
    ) -> EdgeType:
        """Classify node to determine edge type."""
        degree = self.G.degree(node)

        if degree == 1:
            return (
                EdgeType.DOOR_OPENING if node in self.door_nodes else EdgeType.FREE_EDGE
            )

        is_exterior = self._is_edge_on_exterior(wall_edge_u, wall_edge_v)

        if is_exterior:
            neighbor_edges = list(self.G.edges(node))
            exterior_count = sum(
                1 for u, v in neighbor_edges if self._is_edge_on_exterior(u, v)
            )

            if degree > 2 and exterior_count < degree:
                return EdgeType.INTERIOR_JOINT
            return EdgeType.EXTERIOR_JOINT

        return EdgeType.INTERIOR_JOINT

    def analyze_wall(
        self, u: Point, v: Point, wall_id: str = ""
    ) -> WallOverhangConstraints:
        """Analyze overhang constraints for a wall."""
        u = normalize_point(u)
        v = normalize_point(v)

        left_type = self._classify_node(u, u, v)
        right_type = self._classify_node(v, u, v)

        left_inward = self._get_inward_direction_at_node(u, u, v)
        right_inward = self._get_inward_direction_at_node(v, u, v)

        left_neighbors = [n for n in self.G.neighbors(u) if n != v]
        right_neighbors = [n for n in self.G.neighbors(v) if n != u]

        left_constraint = EdgeConstraint(
            edge_type=left_type,
            max_overhang_mm=OVERHANG_LIMITS[left_type],
            inward_direction=left_inward,
            neighbor_node=left_neighbors[0] if left_neighbors else None,
        )

        right_constraint = EdgeConstraint(
            edge_type=right_type,
            max_overhang_mm=OVERHANG_LIMITS[right_type],
            inward_direction=right_inward,
            neighbor_node=right_neighbors[0] if right_neighbors else None,
        )

        return WallOverhangConstraints(
            wall_id=wall_id, left_edge=left_constraint, right_edge=right_constraint
        )

    def get_all_wall_constraints(self) -> Dict[str, WallOverhangConstraints]:
        """Analyze all walls in graph."""
        constraints = {}

        for u, v, data in self.G.edges(data=True):
            wall_id = data.get("wall_id", data.get("original_wall_id", f"{u}-{v}"))
            constraints[wall_id] = self.analyze_wall(u, v, wall_id)

        return constraints

    def visualize_debug_info(self) -> str:
        """Return debug info about contour and doors."""
        lines = [
            "=== Overhang Analyzer Debug ===",
            f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges",
            f"Exterior edges: {len(self.exterior_edges)}",
            f"Door openings: {len(self.door_openings)}",
        ]

        if self.door_openings:
            lines.append("\nDoor openings:")
            for n1, n2 in self.door_openings:
                dist = point_distance(n1, n2)
                lines.append(f"  {n1} <-> {n2}: {dist:.0f}mm")

        lines.append("\nExterior edges with inward normals:")
        for (u, v), normal in list(self.exterior_inward_normals.items())[:10]:
            lines.append(f"  {u} -> {v}: inward=({normal[0]:.2f}, {normal[1]:.2f})")
        if len(self.exterior_inward_normals) > 10:
            lines.append(f"  ... and {len(self.exterior_inward_normals) - 10} more")

        return "\n".join(lines)
