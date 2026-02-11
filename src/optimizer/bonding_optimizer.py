from typing import Dict, List, Tuple, Optional, Set, Any
from itertools import product
from dataclasses import dataclass
import numpy as np
import networkx as nx

from src.builder.structures import WallInstance
from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner


@dataclass
class WallResult:
    """Result of RL inference for a single wall."""
    wall_id: int
    grid: np.ndarray
    instances: Dict[str, Dict]
    reward: float
    quality_score: float


@dataclass
class OptimizationResult:
    """Result of bonding optimization."""
    bonding_assignments: Dict[Tuple[int, int], int]  # (wall_i, wall_j) -> bonding_type
    wall_results: Dict[int, WallResult]  # wall_id -> result
    total_score: float
    num_rl_calls: int


class BondingOptimizer:
    """
    Optimizes bonding patterns at wall joints.

    For each joint between walls, chooses bonding type (0 or 1) that
    maximizes total layout quality across all walls.
    """

    def __init__(
        self,
        runner: ModelRunner,
        context_builder: ContextBuilder,
        beam_width: int = 5,
    ):
        """
        Args:
            runner: ModelRunner for RL inference
            context_builder: ContextBuilder instance (created if None)
            beam_width: Width for beam search (used when graph has cycles)
        """
        self.runner = runner
        self.context_builder = context_builder
        self.beam_width = beam_width
        self.rl_call_count = 0

    def optimize(
        self,
        walls: List[WallInstance],
        adjacency: Dict[int, List[int]],
    ) -> OptimizationResult:
        """
        Find optimal bonding patterns for all wall joints.

        Args:
            walls: List of walls (with .id attribute)
            adjacency: Dict mapping wall_id -> list of adjacent wall_ids
                       Order matters: adjacency[i] = [left_neighbor, right_neighbor]

        Returns:
            OptimizationResult with assignments and layouts
        """
        self.rl_call_count = 0

        # Build graph
        G = self._build_graph(walls, adjacency)

        # Check for cycles
        if self._has_cycles(G):
            return self._beam_search(walls, adjacency, G)
        else:
            return self._tree_dp(walls, adjacency, G)

    def optimize_chain(
        self,
        walls: List[WallInstance],
    ) -> OptimizationResult:
        """
        Optimize for a simple chain of walls (W0 -- W1 -- W2 -- ...).

        Args:
            walls: Ordered list of walls forming a chain

        Returns:
            OptimizationResult
        """
        # Build adjacency for chain
        adjacency = {}
        for i, wall in enumerate(walls):
            neighbors = []
            if i > 0:
                neighbors.append(walls[i - 1].id)
            if i < len(walls) - 1:
                neighbors.append(walls[i + 1].id)
            adjacency[wall.id] = neighbors

        return self.optimize(walls, adjacency)

    def _build_graph(
        self,
        walls: List[WallInstance],
        adjacency: Dict[int, List[int]],
    ) -> nx.Graph:
        """Build NetworkX graph from adjacency."""
        G = nx.Graph()

        wall_ids = [w.id for w in walls]
        G.add_nodes_from(wall_ids)

        for wall_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                if not G.has_edge(wall_id, neighbor_id):
                    G.add_edge(wall_id, neighbor_id)

        return G

    def _has_cycles(self, G: nx.Graph) -> bool:
        return len(G.edges()) >= len(G.nodes())

    def _tree_dp(
        self,
        walls: List[WallInstance],
        adjacency: Dict[int, List[int]],
        G: nx.Graph,
    ) -> OptimizationResult:
        """
        DP on tree: exact solution in O(N * 2^degree).

        For each node, we compute best score for each possible incoming bonding type.
        """
        wall_map = {w.id: w for w in walls}

        # Choose root (node with max degree)
        root = max(G.nodes(), key=lambda n: G.degree(n))

        # Memoization: dp[node][incoming_bonding] = (score, assignments, results)
        dp: Dict[Tuple[int, Optional[int]], Tuple[float, Dict, Dict]] = {}

        def solve(
            node: int,
            parent: Optional[int],
            incoming_bonding: Optional[int],
        ) -> Tuple[float, Dict[Tuple[int, int], int], Dict[int, WallResult]]:
            """
            Solve for subtree rooted at node.

            Args:
                node: Current wall id
                parent: Parent wall id (None for root)
                incoming_bonding: Bonding type from parent edge (None for root)

            Returns:
                (best_score, bonding_assignments, wall_results)
            """
            cache_key = (node, incoming_bonding)
            if cache_key in dp:
                return dp[cache_key]

            children = [n for n in G.neighbors(node) if n != parent]
            wall = wall_map[node]

            # Get neighbor walls for thickness info
            left_wall = wall_map.get(parent) if parent else None

            if not children:
                # Leaf node
                right_wall = None
                bonding_right = None

                result = self._run_inference(
                    wall, left_wall, right_wall,
                    incoming_bonding, bonding_right
                )

                dp[cache_key] = (result.quality_score, {}, {node: result})
                return dp[cache_key]

            best_score = float('-inf')
            best_assignments: Dict[Tuple[int, int], int] = {}
            best_results: Dict[int, WallResult] = {}

            # Try all combinations of outgoing bonding types
            for outgoing in product([0, 1], repeat=len(children)):

                # For simplicity, use first child as "right" neighbor
                right_wall = wall_map.get(children[0]) if children else None
                bonding_right = outgoing[0] if children else None

                # Run RL for current wall
                result = self._run_inference(
                    wall, left_wall, right_wall,
                    incoming_bonding, bonding_right
                )
                wall_score = result.quality_score

                # Recursively solve for children
                children_score = 0.0
                children_assignments: Dict[Tuple[int, int], int] = {}
                children_results: Dict[int, WallResult] = {node: result}

                for child, out_bond in zip(children, outgoing):
                    c_score, c_assign, c_results = solve(child, node, out_bond)
                    children_score += c_score
                    children_assignments.update(c_assign)
                    children_assignments[(node, child)] = out_bond
                    children_results.update(c_results)

                total = wall_score + children_score

                if total > best_score:
                    best_score = total
                    best_assignments = children_assignments
                    best_results = children_results

            dp[cache_key] = (best_score, best_assignments, best_results)
            return dp[cache_key]

        # Try both options for root (no incoming bonding, but may have outgoing)
        best_overall = (float('-inf'), {}, {})

        best_overall = solve(root, None, None)

        return OptimizationResult(
            bonding_assignments=best_overall[1],
            wall_results=best_overall[2],
            total_score=best_overall[0],
            num_rl_calls=self.rl_call_count,
        )

    def _beam_search(
        self,
        walls: List[WallInstance],
        adjacency: Dict[int, List[int]],
        G: nx.Graph,
    ) -> OptimizationResult:
        """
        Beam search for graphs with cycles.

        Processes walls in BFS order, keeping top-k partial solutions.
        """
        wall_map = {w.id: w for w in walls}

        # BFS order from highest-degree node
        start = max(G.nodes(), key=lambda n: G.degree(n))
        order = list(nx.bfs_tree(G, start).nodes())

        # beam: List[(assignments, score, results)]
        beam: List[Tuple[Dict, float, Dict]] = [({}, 0.0, {})]

        processed_joints: Set[Tuple[int, int]] = set()

        for wall_id in order:
            wall = wall_map[wall_id]
            neighbors = list(G.neighbors(wall_id))

            candidates = []

            for assignments, score, results in beam:
                # Find joints that need decisions
                undecided_joints = []
                for neighbor in neighbors:
                    joint = tuple(sorted([wall_id, neighbor]))
                    if joint not in assignments and joint not in processed_joints:
                        undecided_joints.append((neighbor, joint))

                if not undecided_joints:
                    # All joints decided, just run inference
                    left_wall, right_wall = self._get_neighbors(
                        wall_id, neighbors, wall_map
                    )
                    bonding_left = self._get_bonding(
                        assignments, wall_id, neighbors[0] if neighbors else None
                    )
                    bonding_right = self._get_bonding(
                        assignments, wall_id, neighbors[1] if len(neighbors) > 1 else None
                    )

                    result = self._run_inference(
                        wall, left_wall, right_wall,
                        bonding_left, bonding_right
                    )

                    new_results = {**results, wall_id: result}
                    candidates.append((assignments.copy(), score + result.quality_score, new_results))
                else:
                    # Try all bonding combinations for undecided joints
                    for bondings in product([0, 1], repeat=len(undecided_joints)):
                        new_assignments = assignments.copy()
                        for (neighbor, joint), bonding in zip(undecided_joints, bondings):
                            new_assignments[joint] = bonding

                        left_wall, right_wall = self._get_neighbors(
                            wall_id, neighbors, wall_map
                        )
                        bonding_left = self._get_bonding(
                            new_assignments, wall_id, neighbors[0] if neighbors else None
                        )
                        bonding_right = self._get_bonding(
                            new_assignments, wall_id, neighbors[1] if len(neighbors) > 1 else None
                        )

                        result = self._run_inference(
                            wall, left_wall, right_wall,
                            bonding_left, bonding_right
                        )

                        new_results = {**results, wall_id: result}
                        candidates.append((
                            new_assignments,
                            score + result.quality_score,
                            new_results
                        ))

                        # Mark joints as processed
                        for _, joint in undecided_joints:
                            processed_joints.add(joint)

            # Keep top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]

        best = beam[0]
        return OptimizationResult(
            bonding_assignments=best[0],
            wall_results=best[2],
            total_score=best[1],
            num_rl_calls=self.rl_call_count,
        )

    def _get_neighbors(
        self,
        wall_id: int,
        neighbor_ids: List[int],
        wall_map: Dict[int, WallInstance],
    ) -> Tuple[Optional[WallInstance], Optional[WallInstance]]:
        """Get left and right neighbor walls."""
        left = wall_map.get(neighbor_ids[0]) if neighbor_ids else None
        right = wall_map.get(neighbor_ids[1]) if len(neighbor_ids) > 1 else None
        return left, right

    def _get_bonding(
        self,
        assignments: Dict[Tuple[int, int], int],
        wall_id: int,
        neighbor_id: Optional[int],
    ) -> Optional[int]:
        """Get bonding type for a joint from assignments."""
        if neighbor_id is None:
            return None
        joint = tuple(sorted([wall_id, neighbor_id]))
        return assignments.get(joint)

    def _run_inference(
        self,
        wall: WallInstance,
        left_wall: Optional[WallInstance],
        right_wall: Optional[WallInstance],
        bonding_left: Optional[int],
        bonding_right: Optional[int],
    ) -> WallResult:
        """
        Run RL inference for a single wall with given constraints.
        """
        self.rl_call_count += 1

        # Build context grid with specified bonding
        context_grid = self.context_builder.build_grid_with_bonding(
            wall=wall,
            left_wall=left_wall,
            right_wall=right_wall,
            bonding_left=bonding_left,
            bonding_right=bonding_right,
        )

        # Create fake walls list and context_data for ModelRunner
        # This is a workaround until we refactor ModelRunner
        walls_for_runner = [wall]
        context_data = {
            "walls": walls_for_runner,
            "current_idx": 0,
            "bonding_left": bonding_left,
            "bonding_right": bonding_right,
            "left_wall": left_wall,
            "right_wall": right_wall,
        }

        result = self.runner.run(
            wall=wall,
            context_builder=self.context_builder,
            context_data=context_data,
        )

        quality = self._compute_quality(result)

        return WallResult(
            wall_id=wall.id,
            grid=result.get("grid"),
            instances=result.get("instances"),
            reward=result.get("reward", 0.0),
            quality_score=quality,
        )

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """
        Compute quality score for a layout.

        Higher is better. Prefers:
        - More FBS blocks (less monolith)
        - Fewer seams
        - Larger blocks
        """
        grid = result.get("grid")
        instances = result.get("instances", {})

        if grid is None or not instances:
            return 0.0

        # Count cells by type
        total_cells = np.sum(grid > 0)
        if total_cells == 0:
            return 0.0

        # Monolith cells (type_id = 0)
        monolith_cells = 0
        fbs_cells = 0
        seam_count = 0
        block_count = 0

        for inst in instances.values():
            type_id = inst.get("type_id", 0)
            length = inst.get("end", 0) - inst.get("start", 0)

            if type_id == 0:
                monolith_cells += length
            else:
                fbs_cells += length
                block_count += 1
                seam_count += 1  # Each block adds one seam (simplified)

        # Normalize
        fbs_ratio = fbs_cells / total_cells if total_cells > 0 else 0
        monolith_ratio = monolith_cells / total_cells if total_cells > 0 else 0

        # Weighted score
        score = (
            100 * fbs_ratio           # Reward FBS usage
            - 50 * monolith_ratio     # Penalize monolith
            - 0.5 * seam_count        # Penalize many seams
            + result.get("reward", 0) / 10  # Include RL reward
        )

        return score
