from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math
import networkx as nx

from .planer import (
    normalize_point,
    process_intersections,
    process_t_joints,
    choose_start_node,
    traverse_walls,
    Point,
)
from src.builder.structures import WallInstance, GRID_STEP


@dataclass
class WallData:
    """Raw wall data from external source (e.g., Revit)."""
    wall_id: int
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    height: int = 1800
    weight: int = 300 

    @property
    def length(self) -> int:
        """Wall length in mm."""
        dx = self.x_end - self.x_start
        dy = self.y_end - self.y_start
        return int(math.sqrt(dx * dx + dy * dy))

    @property
    def start_point(self) -> Point:
        return normalize_point((self.x_start, self.y_start))

    @property
    def end_point(self) -> Point:
        return normalize_point((self.x_end, self.y_end))


@dataclass
class TraversalItem:
    """Single item in wall traversal order."""
    wall_id: int
    original_wall_id: int
    start_point: Point
    end_point: Point
    order_index: int


class WallPlanner:
    """
    Plans wall construction order and builds adjacency graph.

    Usage:
        planner = WallPlanner()
        planner.add_walls(wall_data_list)
        planner.process()

        order = planner.get_traversal_order()
        adjacency = planner.get_adjacency()
        walls = planner.get_wall_instances()
    """

    def __init__(self, grid_step: int = GRID_STEP):
        self.grid_step = grid_step
        self.wall_data: Dict[int, WallData] = {}
        self.graph: Optional[nx.Graph] = None
        self.processed_graph: Optional[nx.Graph] = None
        self.traversal: List[TraversalItem] = []

    def add_wall(self, wall: WallData) -> None:
        """Add a single wall."""
        self.wall_data[wall.wall_id] = wall

    def add_walls(self, walls: List[WallData]) -> None:
        """Add multiple walls."""
        for wall in walls:
            self.add_wall(wall)

    def add_walls_from_coords(
        self,
        x_start: List[float],
        y_start: List[float],
        x_end: List[float],
        y_end: List[float],
        wall_ids: List[int],
        heights: Optional[List[int]] = None,
        weights: Optional[List[int]] = None,
    ) -> None:
        """Add walls from coordinate arrays (like in planer.py)."""
        n = len(wall_ids)
        heights = heights or [1800] * n
        weights = weights or [300] * n

        for i in range(n):
            wall = WallData(
                wall_id=wall_ids[i],
                x_start=x_start[i],
                y_start=y_start[i],
                x_end=x_end[i],
                y_end=y_end[i],
                height=heights[i],
                weight=weights[i],
            )
            self.add_wall(wall)

    def process(self) -> None:
        """Process walls: build graph, find intersections, compute traversal."""
        if not self.wall_data:
            raise ValueError("No walls added")

        # Build initial graph
        self.graph = nx.Graph()
        for wall in self.wall_data.values():
            self.graph.add_edge(
                wall.start_point,
                wall.end_point,
                wall_id=str(wall.wall_id),
                original_wall_id=str(wall.wall_id),
            )

        # Process intersections and T-joints
        g_cross = process_intersections(self.graph)
        self.processed_graph = process_t_joints(g_cross)

        # Compute traversal order
        start_node = choose_start_node(self.processed_graph)
        raw_traversal = traverse_walls(self.processed_graph, start_node)

        # Convert to TraversalItem list
        self.traversal = []
        seen_originals: Set[int] = set()

        for idx, ((u, v), orig_id_str) in enumerate(raw_traversal):
            orig_id = int(orig_id_str)
            if orig_id not in seen_originals:
                seen_originals.add(orig_id)
                self.traversal.append(TraversalItem(
                    wall_id=orig_id,
                    original_wall_id=orig_id,
                    start_point=u,
                    end_point=v,
                    order_index=len(self.traversal),
                ))

    def get_traversal_order(self) -> List[int]:
        """Get wall IDs in construction order."""
        return [item.wall_id for item in self.traversal]

    def get_adjacency(self) -> Dict[int, List[int]]:
        """
        Get adjacency dict for BondingOptimizer.

        Returns:
            Dict mapping wall_id -> [neighbor_ids] in traversal order
        """
        if not self.traversal:
            return {}

        # Build adjacency from traversal order (chain assumption)
        adjacency: Dict[int, List[int]] = {}
        order = self.get_traversal_order()

        for i, wall_id in enumerate(order):
            neighbors = []
            if i > 0:
                neighbors.append(order[i - 1])  # Previous wall
            if i < len(order) - 1:
                neighbors.append(order[i + 1])  # Next wall
            adjacency[wall_id] = neighbors

        return adjacency

    def get_adjacency_from_graph(self) -> Dict[int, List[int]]:
        """
        Get adjacency from actual graph structure (handles branches).

        More accurate for T-joints and complex graphs.
        """
        if self.processed_graph is None:
            return {}

        # Map points to original wall IDs
        point_to_walls: Dict[Point, Set[int]] = {}

        for wall in self.wall_data.values():
            for pt in [wall.start_point, wall.end_point]:
                if pt not in point_to_walls:
                    point_to_walls[pt] = set()
                point_to_walls[pt].add(wall.wall_id)

        # Also include points from processed graph
        for u, v, data in self.processed_graph.edges(data=True):
            orig_id = int(data.get('original_wall_id', data['wall_id']))
            for pt in [u, v]:
                if pt not in point_to_walls:
                    point_to_walls[pt] = set()
                point_to_walls[pt].add(orig_id)

        # Build adjacency: walls sharing a point are adjacent
        adjacency: Dict[int, List[int]] = {w.wall_id: [] for w in self.wall_data.values()}

        for pt, wall_ids in point_to_walls.items():
            wall_list = list(wall_ids)
            for i, w1 in enumerate(wall_list):
                for w2 in wall_list[i + 1:]:
                    if w2 not in adjacency.get(w1, []):
                        adjacency.setdefault(w1, []).append(w2)
                    if w1 not in adjacency.get(w2, []):
                        adjacency.setdefault(w2, []).append(w1)

        return adjacency

    def get_wall_instances(self) -> List[WallInstance]:
        """
        Get WallInstance objects in traversal order.

        Returns:
            List of WallInstance ready for BondingOptimizer
        """
        order = self.get_traversal_order()
        instances = []

        for wall_id in order:
            wall_data = self.wall_data.get(wall_id)
            if wall_data:
                instance = WallInstance(
                    id=wall_data.wall_id,
                    length=wall_data.length,
                    height=wall_data.height,
                    weight=wall_data.weight,
                    grid_step=self.grid_step,
                )
                instances.append(instance)

        return instances

    def get_stats(self) -> Dict:
        """Get statistics about the wall graph."""
        return {
            "total_walls": len(self.wall_data),
            "traversal_length": len(self.traversal),
            "graph_nodes": self.processed_graph.number_of_nodes() if self.processed_graph else 0,
            "graph_edges": self.processed_graph.number_of_edges() if self.processed_graph else 0,
            "connected_components": nx.number_connected_components(self.processed_graph) if self.processed_graph else 0,
            "has_cycles": self._has_cycles(),
        }

    def _has_cycles(self) -> bool:
        """Check if processed graph has cycles."""
        if self.processed_graph is None:
            return False
        return self.processed_graph.number_of_edges() >= self.processed_graph.number_of_nodes()
