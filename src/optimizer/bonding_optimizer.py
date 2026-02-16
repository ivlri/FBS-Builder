from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.builder.structures import WallInstance
from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner


@dataclass
class WallResult:
    wall_id: int
    grid: np.ndarray
    instances: Dict[str, Dict]
    reward: float
    quality_score: float


@dataclass
class OptimizationResult:
    bonding_assignments: Dict[Tuple[int, int], int]
    wall_results: Dict[int, WallResult]
    total_score: float
    num_rl_calls: int


class BondingOptimizer:
    """Optimizes bonding patterns at wall joints using RL inference."""

    def __init__(
        self, runner: ModelRunner, context_builder: ContextBuilder, beam_width: int = 5
    ):
        self.runner = runner
        self.context_builder = context_builder
        self.beam_width = beam_width
        self.rl_call_count = 0

    def optimize(
        self, walls: List[WallInstance], adjacency: Dict[int, List[int]]
    ) -> OptimizationResult:
        """Find optimal bonding patterns for all wall joints."""
        self.rl_call_count = 0
        G = self._build_graph(walls, adjacency)

        if self._has_cycles(G):
            return self._beam_search(walls, adjacency, G)
        return self._tree_dp(walls, adjacency, G)

    def optimize_chain(self, walls: List[WallInstance]) -> OptimizationResult:
        """Optimize simple chain (W0 -- W1 -- W2 -- ...)."""
        if len(walls) == 0:
            return OptimizationResult({}, {}, 0.0, 0)

        if len(walls) == 1:
            result = self._run_inference(walls[0], None, None, None, None)
            return OptimizationResult(
                {}, {walls[0].id: result}, result.quality_score, 1
            )

        n_joints = len(walls) - 1
        best_result = None
        total_rl_calls = 0

        for bonding_combo in product([0, 1], repeat=n_joints):
            result = self._process_chain(walls, bonding_combo)
            total_rl_calls += len(walls)

            if best_result is None or result.total_score > best_result.total_score:
                best_result = result

        return OptimizationResult(
            bonding_assignments=best_result.bonding_assignments,
            wall_results=best_result.wall_results,
            total_score=best_result.total_score,
            num_rl_calls=total_rl_calls,
        )

    def _process_chain(
        self, walls: List[WallInstance], bonding_combo: Tuple[int, ...]
    ) -> OptimizationResult:
        """Process chain with given bonding combination."""
        wall_results: Dict[int, WallResult] = {}
        bonding_assignments: Dict[Tuple[int, int], int] = {}
        total_score = 0.0

        for i, bonding in enumerate(bonding_combo):
            bonding_assignments[(walls[i].id, walls[i + 1].id)] = bonding

        left_occupied = None
        for i, wall in enumerate(walls):
            bonding_left, bonding_right = self._get_wall_bondings(
                i, walls, bonding_assignments
            )
            left_wall = walls[i - 1] if i > 0 else None
            right_wall = walls[i + 1] if i < len(walls) - 1 else None

            result = self._run_inference(
                wall, left_wall, right_wall, bonding_left, bonding_right, left_occupied
            )
            wall_results[wall.id] = result
            total_score += result.quality_score

            left_occupied = self._extract_edge_for_next(i, walls, result)

        return OptimizationResult(
            bonding_assignments, wall_results, total_score, len(walls)
        )

    def _get_wall_bondings(
        self, i: int, walls: List[WallInstance], assignments: Dict[Tuple[int, int], int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Get bonding types for wall at index i."""
        bonding_left = None
        bonding_right = None

        if i > 0:
            joint = (walls[i - 1].id, walls[i].id)
            bonding = assignments.get(joint)
            bonding_left = 1 - bonding if bonding is not None else None

        if i < len(walls) - 1:
            joint = (walls[i].id, walls[i + 1].id)
            bonding_right = assignments.get(joint)

        return bonding_left, bonding_right

    def _extract_edge_for_next(
        self, i: int, walls: List[WallInstance], result: WallResult
    ) -> Optional[np.ndarray]:
        """Extract right edge occupied cells for next wall."""
        if i < len(walls) - 1 and result.grid is not None:
            next_wall_thickness = walls[i + 1].weight
            width_cells = next_wall_thickness // self.context_builder.grid_step
            return self.context_builder.extract_edge_occupied(
                result.grid, "right", width_cells
            )
        return None

    def _build_graph(
        self, walls: List[WallInstance], adjacency: Dict[int, List[int]]
    ) -> nx.Graph:
        """Build NetworkX graph from adjacency."""
        G = nx.Graph()
        G.add_nodes_from([w.id for w in walls])

        for wall_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                if not G.has_edge(wall_id, neighbor_id):
                    G.add_edge(wall_id, neighbor_id)

        return G

    def _has_cycles(self, G: nx.Graph) -> bool:
        return len(G.edges()) >= len(G.nodes())

    def _tree_dp(
        self, walls: List[WallInstance], adjacency: Dict[int, List[int]], G: nx.Graph
    ) -> OptimizationResult:
        """DP on tree: O(N * 2^degree)."""
        wall_map = {w.id: w for w in walls}
        root = max(G.nodes(), key=lambda n: G.degree(n))
        dp: Dict[Tuple[int, Optional[int]], Tuple[float, Dict, Dict]] = {}

        def solve(node: int, parent: Optional[int], incoming_bonding: Optional[int]):
            cache_key = (node, incoming_bonding)
            if cache_key in dp:
                return dp[cache_key]

            children = [n for n in G.neighbors(node) if n != parent]
            wall = wall_map[node]
            left_wall = wall_map.get(parent) if parent else None

            if not children:
                result = self._run_inference(
                    wall, left_wall, None, incoming_bonding, None
                )
                dp[cache_key] = (result.quality_score, {}, {node: result})
                return dp[cache_key]

            best = self._find_best_child_combination(
                wall, left_wall, incoming_bonding, children, wall_map, solve, node
            )
            dp[cache_key] = best
            return best

        best_overall = solve(root, None, None)
        return OptimizationResult(
            best_overall[1], best_overall[2], best_overall[0], self.rl_call_count
        )

    def _find_best_child_combination(
        self, wall, left_wall, incoming_bonding, children, wall_map, solve, node
    ):
        """Find best combination of outgoing bonding types for children."""
        best_score = float("-inf")
        best_assignments: Dict[Tuple[int, int], int] = {}
        best_results: Dict[int, WallResult] = {}

        for outgoing in product([0, 1], repeat=len(children)):
            right_wall = wall_map.get(children[0]) if children else None
            bonding_right = outgoing[0] if children else None

            result = self._run_inference(
                wall, left_wall, right_wall, incoming_bonding, bonding_right
            )
            total, assignments, results = self._aggregate_children(
                result, children, outgoing, node, solve
            )

            if total > best_score:
                best_score = total
                best_assignments = assignments
                best_results = results

        return best_score, best_assignments, best_results

    def _aggregate_children(self, wall_result, children, outgoing, node, solve):
        """Aggregate scores from children."""
        total = wall_result.quality_score
        assignments: Dict[Tuple[int, int], int] = {}
        results: Dict[int, WallResult] = {node: wall_result}

        for child, out_bond in zip(children, outgoing):
            c_score, c_assign, c_results = solve(child, node, out_bond)
            total += c_score
            assignments.update(c_assign)
            assignments[(node, child)] = out_bond
            results.update(c_results)

        return total, assignments, results

    def _beam_search(
        self, walls: List[WallInstance], adjacency: Dict[int, List[int]], G: nx.Graph
    ) -> OptimizationResult:
        """Beam search for graphs with cycles."""
        wall_map = {w.id: w for w in walls}
        start = max(G.nodes(), key=lambda n: G.degree(n))
        order = list(nx.bfs_tree(G, start).nodes())

        beam: List[Tuple[Dict, float, Dict]] = [({}, 0.0, {})]
        processed_joints: Set[Tuple[int, int]] = set()

        for wall_id in order:
            beam = self._expand_beam_for_wall(
                wall_id, G, wall_map, beam, processed_joints
            )

        best = beam[0]
        return OptimizationResult(best[0], best[2], best[1], self.rl_call_count)

    def _expand_beam_for_wall(self, wall_id, G, wall_map, beam, processed_joints):
        """Expand beam for single wall."""
        wall = wall_map[wall_id]
        neighbors = list(G.neighbors(wall_id))
        candidates = []

        for assignments, score, results in beam:
            undecided = self._find_undecided_joints(
                wall_id, neighbors, assignments, processed_joints
            )

            if not undecided:
                candidates.extend(
                    self._handle_decided_wall(
                        wall_id, wall, neighbors, assignments, score, results, wall_map
                    )
                )
            else:
                candidates.extend(
                    self._handle_undecided_wall(
                        wall_id,
                        wall,
                        neighbors,
                        assignments,
                        score,
                        results,
                        undecided,
                        wall_map,
                        processed_joints,
                    )
                )

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.beam_width]

    def _find_undecided_joints(self, wall_id, neighbors, assignments, processed_joints):
        """Find joints that need bonding decisions."""
        undecided = []
        for neighbor in neighbors:
            joint = tuple(sorted([wall_id, neighbor]))
            if joint not in assignments and joint not in processed_joints:
                undecided.append((neighbor, joint))
        return undecided

    def _handle_decided_wall(
        self, wall_id, wall, neighbors, assignments, score, results, wall_map
    ):
        """Handle wall where all joints are decided."""
        left_wall, right_wall = self._get_neighbors(wall_id, neighbors, wall_map)
        bonding_left = self._get_bonding(
            assignments, wall_id, neighbors[0] if neighbors else None
        )
        bonding_right = self._get_bonding(
            assignments, wall_id, neighbors[1] if len(neighbors) > 1 else None
        )

        result = self._run_inference(
            wall, left_wall, right_wall, bonding_left, bonding_right
        )
        new_results = {**results, wall_id: result}
        return [(assignments.copy(), score + result.quality_score, new_results)]

    def _handle_undecided_wall(
        self,
        wall_id,
        wall,
        neighbors,
        assignments,
        score,
        results,
        undecided,
        wall_map,
        processed_joints,
    ):
        """Handle wall with undecided joints."""
        candidates = []

        for bondings in product([0, 1], repeat=len(undecided)):
            new_assignments = assignments.copy()
            for (neighbor, joint), bonding in zip(undecided, bondings):
                new_assignments[joint] = bonding

            left_wall, right_wall = self._get_neighbors(wall_id, neighbors, wall_map)
            bonding_left = self._get_bonding(
                new_assignments, wall_id, neighbors[0] if neighbors else None
            )
            bonding_right = self._get_bonding(
                new_assignments, wall_id, neighbors[1] if len(neighbors) > 1 else None
            )

            result = self._run_inference(
                wall, left_wall, right_wall, bonding_left, bonding_right
            )
            new_results = {**results, wall_id: result}
            candidates.append(
                (new_assignments, score + result.quality_score, new_results)
            )

            for _, joint in undecided:
                processed_joints.add(joint)

        return candidates

    def _get_neighbors(
        self, wall_id: int, neighbor_ids: List[int], wall_map: Dict[int, WallInstance]
    ) -> Tuple[Optional[WallInstance], Optional[WallInstance]]:
        """Get left and right neighbor walls."""
        left = wall_map.get(neighbor_ids[0]) if neighbor_ids else None
        right = wall_map.get(neighbor_ids[1]) if len(neighbor_ids) > 1 else None
        return left, right

    def _get_bonding(
        self, assignments: Dict, wall_id: int, neighbor_id: Optional[int]
    ) -> Optional[int]:
        """Get bonding type for joint."""
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
        left_occupied: Optional[np.ndarray] = None,
    ) -> WallResult:
        """Run RL inference for single wall."""
        self.rl_call_count += 1

        context_data = {
            "walls": [wall],
            "current_idx": 0,
            "bonding_left": bonding_left,
            "bonding_right": bonding_right,
            "left_wall": left_wall,
            "right_wall": right_wall,
        }

        result = self.runner.run(
            wall=wall, context_builder=self.context_builder, context_data=context_data
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
        """Compute quality: prefer FBS, penalize monolith and seams."""
        grid = result.get("grid")
        instances = result.get("instances", {})

        if grid is None or not instances:
            return 0.0

        total_cells = np.sum(grid > 0)
        if total_cells == 0:
            return 0.0

        monolith_cells, fbs_cells, seam_count = 0, 0, 0

        for inst in instances.values():
            type_id = inst.get("type_id", 0)
            length = inst.get("end", 0) - inst.get("start", 0)

            if type_id == 0:
                monolith_cells += length
            else:
                fbs_cells += length
                seam_count += 1

        fbs_ratio = fbs_cells / total_cells
        monolith_ratio = monolith_cells / total_cells

        return (
            100 * fbs_ratio
            - 50 * monolith_ratio
            - 0.5 * seam_count
            + result.get("reward", 0) / 10
        )
