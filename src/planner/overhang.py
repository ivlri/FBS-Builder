"""
Overhang detection system for FBS walls.

Determines maximum allowed block overhang at each wall edge based on:
- Exterior contour → 0mm (forbidden)
- Interior joint → up to 500mm
- Door opening (800-1100mm gap) → up to 200mm
- Monolith → always forbidden (regardless of edge type)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from collections import defaultdict

Point = Tuple[float, float]


class EdgeType(Enum):
    FREE_EDGE = 0        # Free wall end → 0mm overhang
    DOOR_OPENING = 1     # Door opening → 200mm overhang
    EXTERIOR_JOINT = 2   # Exterior corner/joint → 0mm overhang
    INTERIOR_JOINT = 3   # Interior joint → 500mm overhang


# Overhang limits in mm
OVERHANG_LIMITS = {
    EdgeType.FREE_EDGE: 0,
    EdgeType.DOOR_OPENING: 200,
    EdgeType.EXTERIOR_JOINT: 0,
    EdgeType.INTERIOR_JOINT: 500,
}


@dataclass
class EdgeConstraint:
    """Constraint for one wall edge (left or right)."""
    edge_type: EdgeType
    max_overhang_mm: int
    inward_direction: Optional[Tuple[float, float]] = None  # Normal pointing inward
    neighbor_node: Optional[Point] = None  # Adjacent node if joint


@dataclass
class WallOverhangConstraints:
    """Overhang constraints for a wall."""
    wall_id: str
    left_edge: EdgeConstraint
    right_edge: EdgeConstraint


def normalize_point(p: Point, precision: int = 6) -> Point:
    """Normalize point to avoid float comparison issues."""
    return (round(p[0], precision), round(p[1], precision))


def point_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def cross_product_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """2D cross product (z-component of 3D cross)."""
    return v1[0] * v2[1] - v1[1] * v2[0]


class OverhangAnalyzer:
    """
    Analyzes wall graph to determine overhang constraints.

    Algorithm:
    1. Find exterior contour via clockwise traversal
    2. Identify door openings (degree-1 nodes 800-1100mm apart)
    3. For each wall edge, classify as exterior/interior/door
    """

    DOOR_MIN_MM = 800
    DOOR_MAX_MM = 1100

    def __init__(self, G: nx.Graph):
        """
        Initialize analyzer with wall graph.

        Args:
            G: NetworkX graph where nodes are (x, y) points and edges are walls.
               Edges should have 'wall_id' or 'original_wall_id' attribute.
        """
        self.G = G

        # Exterior contour info
        self.exterior_edges: Set[Tuple[Point, Point]] = set()
        self.exterior_inward_normals: Dict[Tuple[Point, Point], Tuple[float, float]] = {}

        # Door openings: pairs of degree-1 nodes
        self.door_openings: List[Tuple[Point, Point]] = []
        self.door_nodes: Set[Point] = set()

        # Analyze graph
        self._find_exterior_contour()
        self._find_door_openings()

    def _find_leftmost_bottom_node(self) -> Optional[Point]:
        """Find leftmost-bottom node (guaranteed on exterior contour)."""
        if not self.G.nodes():
            return None

        nodes = list(self.G.nodes())
        # Sort by x (ascending), then by y (ascending)
        return min(nodes, key=lambda n: (n[0], n[1]))

    def _get_outgoing_angle(self, current: Point, next_node: Point) -> float:
        """Get angle of edge from current to next_node (in radians, 0 = right)."""
        dx = next_node[0] - current[0]
        dy = next_node[1] - current[1]
        return np.arctan2(dy, dx)

    def _find_rightmost_turn(self, current: Point, prev: Optional[Point]) -> Optional[Point]:
        """
        Find neighbor that makes the rightmost turn (clockwise traversal).

        For clockwise traversal, we want the neighbor that is "most to the right"
        relative to our incoming direction.
        """
        neighbors = list(self.G.neighbors(current))
        if not neighbors:
            return None

        # Remove previous node from candidates (don't go back)
        if prev is not None:
            neighbors = [n for n in neighbors if n != prev]

        if not neighbors:
            return None

        if len(neighbors) == 1:
            return neighbors[0]

        # Compute incoming angle
        if prev is not None:
            incoming_angle = self._get_outgoing_angle(prev, current)
        else:
            # Start: assume we came from the left (angle = pi)
            incoming_angle = np.pi

        # For each neighbor, compute turn angle
        # Rightmost turn = smallest counter-clockwise angle from incoming direction
        def turn_angle(neighbor):
            outgoing = self._get_outgoing_angle(current, neighbor)
            # Angle from incoming to outgoing (positive = counter-clockwise)
            diff = outgoing - incoming_angle
            # Normalize to [0, 2*pi)
            while diff < 0:
                diff += 2 * np.pi
            while diff >= 2 * np.pi:
                diff -= 2 * np.pi
            return diff

        # Smallest turn angle = rightmost turn
        return min(neighbors, key=turn_angle)

    def _find_exterior_contour(self):
        """
        Find exterior contour by clockwise traversal.

        Start from leftmost-bottom node (guaranteed on exterior).
        Traverse clockwise by always taking "rightmost" turn.
        Interior is always on the RIGHT during traversal.
        """
        start = self._find_leftmost_bottom_node()
        if start is None:
            return

        visited_edges: Set[Tuple[Point, Point]] = set()
        current = start
        prev = None

        # Traverse until we return to start
        max_iterations = len(self.G.edges()) * 2 + 10
        for _ in range(max_iterations):
            next_node = self._find_rightmost_turn(current, prev)
            if next_node is None:
                break

            # Record edge as exterior (in traversal direction)
            edge_key = (current, next_node)
            if edge_key in visited_edges:
                # Completed the loop
                break

            visited_edges.add(edge_key)
            self.exterior_edges.add(edge_key)

            # Compute inward normal (interior is on RIGHT)
            # Edge direction: (dx, dy) = next_node - current
            # Right normal: (dy, -dx) points to the right = interior
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
        """
        Find door openings: pairs of degree-1 nodes with distance 800-1100mm.
        """
        # Find all degree-1 nodes
        degree_1_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 1]

        # Check all pairs
        for i, n1 in enumerate(degree_1_nodes):
            for n2 in degree_1_nodes[i+1:]:
                dist = point_distance(n1, n2)
                if self.DOOR_MIN_MM <= dist <= self.DOOR_MAX_MM:
                    self.door_openings.append((n1, n2))
                    self.door_nodes.add(n1)
                    self.door_nodes.add(n2)

    def _is_edge_on_exterior(self, u: Point, v: Point) -> bool:
        """Check if edge is on exterior contour (in either direction)."""
        return (u, v) in self.exterior_edges or (v, u) in self.exterior_edges

    def _get_inward_direction_at_node(self, node: Point, edge_u: Point, edge_v: Point) -> Optional[Tuple[float, float]]:
        """
        Get inward direction at a node based on exterior contour.

        Returns the inward normal if the edge is on exterior, None otherwise.
        """
        if (edge_u, edge_v) in self.exterior_inward_normals:
            return self.exterior_inward_normals[(edge_u, edge_v)]
        if (edge_v, edge_u) in self.exterior_inward_normals:
            # Reverse the normal
            normal = self.exterior_inward_normals[(edge_v, edge_u)]
            return (-normal[0], -normal[1])
        return None

    def _classify_node(self, node: Point, wall_edge_u: Point, wall_edge_v: Point) -> EdgeType:
        """
        Classify a node to determine edge type.

        Args:
            node: The node to classify (one endpoint of the wall)
            wall_edge_u, wall_edge_v: The wall endpoints (edge in graph)
        """
        degree = self.G.degree(node)

        # Degree 1: free edge or door opening
        if degree == 1:
            if node in self.door_nodes:
                return EdgeType.DOOR_OPENING
            return EdgeType.FREE_EDGE

        # Degree >= 2: joint (check if exterior or interior)
        # Check if THIS wall edge is on exterior contour
        is_exterior = self._is_edge_on_exterior(wall_edge_u, wall_edge_v)

        if is_exterior:
            # Edge is on exterior - check direction of neighbor edges at this node
            # If ALL edges at this node are exterior, it's exterior joint
            # If some edges go interior, determine based on angles
            neighbor_edges = list(self.G.edges(node))
            exterior_count = sum(1 for u, v in neighbor_edges if self._is_edge_on_exterior(u, v))

            # Simple heuristic: if node has >2 degree and not all exterior, likely interior joint
            if degree > 2 and exterior_count < degree:
                return EdgeType.INTERIOR_JOINT
            return EdgeType.EXTERIOR_JOINT
        else:
            # Edge is not on exterior contour - interior joint
            return EdgeType.INTERIOR_JOINT

    def analyze_wall(self, u: Point, v: Point, wall_id: str = "") -> WallOverhangConstraints:
        """
        Analyze overhang constraints for a wall.

        Args:
            u: Start point of wall
            v: End point of wall
            wall_id: Optional wall identifier

        Returns:
            WallOverhangConstraints with left and right edge info
        """
        u = normalize_point(u)
        v = normalize_point(v)

        # Classify both endpoints
        left_type = self._classify_node(u, u, v)
        right_type = self._classify_node(v, u, v)

        # Get inward directions
        left_inward = self._get_inward_direction_at_node(u, u, v)
        right_inward = self._get_inward_direction_at_node(v, u, v)

        # Get neighbor nodes for joints
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
            wall_id=wall_id,
            left_edge=left_constraint,
            right_edge=right_constraint,
        )

    def get_all_wall_constraints(self) -> Dict[str, WallOverhangConstraints]:
        """
        Analyze all walls in the graph.

        Returns:
            Dict mapping wall_id to WallOverhangConstraints
        """
        constraints = {}

        for u, v, data in self.G.edges(data=True):
            wall_id = data.get('wall_id', data.get('original_wall_id', f"{u}-{v}"))
            constraints[wall_id] = self.analyze_wall(u, v, wall_id)

        return constraints

    def visualize_debug_info(self) -> str:
        """Return debug info about exterior contour and door openings."""
        lines = []
        lines.append("=== Overhang Analyzer Debug ===")
        lines.append(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        lines.append(f"Exterior edges: {len(self.exterior_edges)}")
        lines.append(f"Door openings: {len(self.door_openings)}")

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
