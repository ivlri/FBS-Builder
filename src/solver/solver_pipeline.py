from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.builder.structures import BLOCK_TYPES, GRID_STEP, WallInstance
from src.planner.overhang import OverhangAnalyzer
from src.solver.solver import BlockType as SolverBlockType
from src.solver.solver import FBSSolver, Opening
from src.solver.solver import WallInstance as SolverWall


@dataclass
class SolverResult:
    wall_id: int
    grid: np.ndarray
    instances: Dict[int, Dict]
    stats: Dict
    blocked_mask: Optional[np.ndarray] = None
    left_overhang_mm: int = 0
    right_overhang_mm: int = 0


@dataclass
class PipelineResult:
    wall_results: Dict[int, SolverResult]
    total_stats: Dict


class SolverPipeline:
    """
    Process chain of walls with chess-pattern bonding.

    Chess-pattern: bonding=0 blocks layers 0,2,4 / bonding=1 blocks layers 1,3,5
    """

    def __init__(
        self,
        grid_step: int = GRID_STEP,
        row_height: int = 300,
        beam_width: int = 8,
    ):
        self.grid_step = grid_step
        self.row_height = row_height
        self.beam_width = beam_width
        self.overhang_analyzer: Optional[OverhangAnalyzer] = None
        self.wall_nodes: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        self.block_types = [
            SolverBlockType(id=bt.id, length=bt.length, height=bt.height)
            for bt in BLOCK_TYPES
        ]

    def set_wall_graph(self, G: nx.Graph, wall_node_mapping: Optional[Dict] = None):
        """Initialize overhang analyzer with wall graph."""
        self.overhang_analyzer = OverhangAnalyzer(G)
        if wall_node_mapping:
            self.wall_nodes = wall_node_mapping

    def get_overhang_constraints(
        self,
        wall_id: int,
        start_node: Optional[Tuple[float, float]] = None,
        end_node: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, int]:
        """Get overhang constraints for a wall. Returns (left_overhang_mm, right_overhang_mm)."""
        if self.overhang_analyzer is None:
            return (0, 0)

        if start_node is None or end_node is None:
            if wall_id in self.wall_nodes:
                start_node, end_node = self.wall_nodes[wall_id]
            else:
                return (0, 0)

        constraints = self.overhang_analyzer.analyze_wall(
            start_node, end_node, str(wall_id)
        )
        return (
            constraints.left_edge.max_overhang_mm,
            constraints.right_edge.max_overhang_mm,
        )

    def solve_chain(
        self,
        walls: List[WallInstance],
        initial_bonding: int = 0,
        openings_map: Optional[Dict[int, List]] = None,
    ) -> PipelineResult:
        """
        Solve chain of connected walls with chess-pattern bonding.

        Args:
            walls: Ordered list of walls (left to right)
            initial_bonding: Starting pattern (0 or 1)
            openings_map: Dict mapping wall_id -> list of Opening objects
        """
        if not walls:
            return PipelineResult({}, {"total_fbs": 0, "total_mono": 0})

        results: Dict[int, SolverResult] = {}
        left_occupied: Optional[np.ndarray] = None
        current_bonding = initial_bonding
        total_fbs, total_mono = 0, 0

        for i, wall in enumerate(walls):
            result = self._process_single_wall(
                wall, i, walls, current_bonding, left_occupied, openings_map
            )

            results[wall.id] = result
            total_fbs += result.stats["fbs_count"]
            total_mono += result.stats["monolith_cells"]

            if i < len(walls) - 1:
                right_wall = walls[i + 1]
                width_cells = right_wall.weight // self.grid_step
                left_occupied = self._extract_edge(result.grid, "right", width_cells)
                current_bonding = 1 - current_bonding
            else:
                left_occupied = None

        return PipelineResult(
            wall_results=results,
            total_stats={
                "total_fbs": total_fbs,
                "total_mono": total_mono,
                "wall_count": len(walls),
            },
        )

    def _process_single_wall(
        self,
        wall: WallInstance,
        index: int,
        walls: List[WallInstance],
        bonding: int,
        left_occupied: Optional[np.ndarray],
        openings_map: Optional[Dict[int, List]],
    ) -> SolverResult:
        """Process single wall with constraints from neighbors."""
        num_rows = wall.height // self.row_height
        right_wall = walls[index + 1] if index < len(walls) - 1 else None
        left_wall = walls[index - 1] if index > 0 else None

        # Build blocked mask
        blocked = self._build_chess_pattern_mask(
            num_rows, wall.num_cells, bonding, left_wall, right_wall
        )

        if left_occupied is not None:
            blocked = self._merge_occupied(blocked, left_occupied)

        solver = self._prepare_solver(wall, blocked, openings_map)

        grid, instances = solver.solve_wall()
        stats = self._compute_stats(instances)

        left_oh, right_oh = self.get_overhang_constraints(wall.id)

        return SolverResult(
            wall_id=wall.id,
            grid=grid,
            instances=instances,
            stats=stats,
            blocked_mask=blocked.copy(),
            left_overhang_mm=left_oh,
            right_overhang_mm=right_oh,
        )

    def _build_chess_pattern_mask(
        self,
        num_rows: int,
        num_cells: int,
        bonding: int,
        left_wall: Optional[WallInstance],
        right_wall: Optional[WallInstance],
    ) -> np.ndarray:
        """Build blocked mask with chess-pattern bonding at wall joints."""
        blocked = np.zeros((num_rows, num_cells), dtype=bool)

        # Block right edge for interlocking with right neighbor
        if right_wall is not None:
            width_cells = right_wall.weight // self.grid_step
            self._apply_bonding_pattern(blocked, bonding, -width_cells, None)

        # Block left edge where neighbor placed blocks
        if left_wall is not None:
            width_cells = left_wall.weight // self.grid_step
            self._apply_bonding_pattern(blocked, bonding, 0, width_cells)

        return blocked

    def _apply_bonding_pattern(
        self, blocked: np.ndarray, bonding: int, start_col: int, end_col: Optional[int]
    ) -> None:
        """Apply chess-pattern: block even/odd layers in specified column range."""
        num_rows = blocked.shape[0]
        for row in range(num_rows):
            layer = row // 2
            if layer % 2 == bonding:
                blocked[row, start_col:end_col] = True

    def _merge_occupied(
        self, blocked: np.ndarray, left_occupied: np.ndarray
    ) -> np.ndarray:
        """Merge occupied cells from neighbor into blocked mask."""
        min_rows = min(blocked.shape[0], left_occupied.shape[0])
        min_cols = min(blocked.shape[1], left_occupied.shape[1])

        for r in range(min_rows):
            for c in range(min_cols):
                if left_occupied[r, c]:
                    blocked[r, c] = True

        return blocked

    def _prepare_solver(
        self,
        wall: WallInstance,
        blocked: np.ndarray,
        openings_map: Optional[Dict[int, List]],
    ) -> FBSSolver:
        """Create and configure solver with constraints."""
        wall_openings = openings_map.get(wall.id) if openings_map else None
        solver_openings = self._convert_openings_list(wall_openings)
        left_oh, right_oh = self.get_overhang_constraints(wall.id)

        solver_wall = SolverWall(length=wall.length, height=wall.height)
        solver = FBSSolver(
            wall=solver_wall,
            block_types=self.block_types,
            openings=solver_openings,
            grid_step=self.grid_step,
            row_height=self.row_height,
            beam_width=self.beam_width,
            left_overhang_mm=left_oh,
            right_overhang_mm=right_oh,
        )

        self._apply_blocked_mask(solver, blocked)
        return solver

    def _apply_blocked_mask(self, solver: FBSSolver, blocked: np.ndarray) -> None:
        """Apply blocked mask to solver grid."""
        rows, cols = blocked.shape
        rows = min(rows, solver.num_rows)
        cols = min(cols, solver.num_cells)

        for r in range(rows):
            for c in range(cols):
                if blocked[r, c]:
                    solver.blocked[r, c] = 1
                    solver.grid[r, c] = -1

    def _convert_openings_list(
        self, openings: Optional[List]
    ) -> Optional[List[Opening]]:
        """Convert Opening objects to solver format."""
        if not openings:
            return None

        return [
            Opening(
                center_x=op.center_x,
                center_y=op.center_y,
                width=op.width,
                height=op.height,
            )
            for op in openings
        ]

    def _extract_edge(
        self, grid: np.ndarray, side: str, width_cells: int
    ) -> np.ndarray:
        """Extract occupied cells from edge. Returns binary mask."""
        edge = grid[:, -width_cells:] if side == "right" else grid[:, :width_cells]
        return edge > 0

    def _compute_stats(self, instances: Dict) -> Dict:
        """Compute block statistics."""
        fbs_count, monolith_cells, fbs_cells = 0, 0, 0

        for inst in instances.values():
            type_id = inst["type_id"]
            length = inst["end_cell"] - inst["start_cell"]

            if type_id == 0:
                monolith_cells += length
            else:
                fbs_cells += length
                fbs_count += 1

        total = monolith_cells + fbs_cells

        return {
            "fbs_count": fbs_count,
            "fbs_cells": fbs_cells,
            "monolith_cells": monolith_cells,
            "fbs_percent": (fbs_cells / total * 100) if total > 0 else 0,
            "monolith_percent": (monolith_cells / total * 100) if total > 0 else 0,
        }


def normalize_instances(instances: Dict) -> Dict:
    """Convert solver instances to common format (same as RL)."""
    normalized = {}
    for inst_id, inst in instances.items():
        normalized[inst_id] = {
            "row": inst["row"],
            "start": inst["start_cell"],
            "end": inst["end_cell"],
            "h_rows": inst.get("h_rows", 1),
            "type_id": inst["type_id"],
        }
    return normalized


def visualize_wall(
    result: SolverResult, wall: WallInstance, grid_step: int = 20
) -> str:
    """
    ASCII visualization of wall with blocks and overhang.

    Legend: X=blocked, .=empty, M=monolith, 2-7=FBS, <=left overhang, >=right overhang
    """
    grid = result.grid
    blocked = result.blocked_mask
    num_rows, num_cells = grid.shape

    # Find overhang from instances
    max_oh_left, max_oh_right = 0, 0
    for inst in result.instances.values():
        start = inst["start_cell"]
        end = inst["end_cell"]
        if start < 0:
            max_oh_left = max(max_oh_left, -start)
        if end > num_cells:
            max_oh_right = max(max_oh_right, end - num_cells)

    # Create extended grid
    offset = max_oh_left
    extended_cells = max_oh_left + num_cells + max_oh_right
    extended_type_grid = np.zeros((num_rows, extended_cells), dtype=int)

    for inst in result.instances.values():
        type_id = inst["type_id"]
        row = inst["row"]
        h_rows = inst.get("h_rows", 1)
        start = max(0, inst["start_cell"] + offset)
        end = min(extended_cells, inst["end_cell"] + offset)
        extended_type_grid[row : row + h_rows, start:end] = (
            type_id if type_id > 0 else -2
        )

    lines = [
        f"Wall {result.wall_id}: {wall.length}mm x {wall.height}mm",
        f"Grid: {num_rows} rows x {num_cells} cells ({grid_step}mm/cell)",
    ]

    if result.left_overhang_mm > 0 or result.right_overhang_mm > 0:
        lines.append(
            f"Overhang limits: L={result.left_overhang_mm}mm R={result.right_overhang_mm}mm"
        )

    lines.append("")

    # Header
    header = "     " + ("|" if max_oh_left > 0 else "")
    for c in range(0, num_cells, 10):
        header += f"{c * grid_step:<10}"
    if max_oh_right > 0:
        header += "|"
    lines.append(header)

    # Rows from top to bottom
    for row in range(num_rows - 1, -1, -1):
        layer = row // 2
        row_str = f"R{row:02d} "

        for c in range(extended_cells):
            real_c = c - offset

            if real_c < 0:
                t = extended_type_grid[row, c]
                row_str += "<" if t != 0 else " "
            elif real_c >= num_cells:
                t = extended_type_grid[row, c]
                row_str += ">" if t != 0 else " "
            else:
                if blocked is not None and blocked[row, real_c]:
                    row_str += "X"
                elif grid[row, real_c] == -1:
                    row_str += "X"
                elif grid[row, real_c] == 0:
                    row_str += "."
                else:
                    t = extended_type_grid[row, c]
                    row_str += "M" if t == -2 else ("." if t == 0 else str(t % 10))

        lines.append(row_str + f" | L{layer}")

    # Blocked zones summary
    if blocked is not None:
        lines.extend(_format_blocked_zones(blocked, num_rows, num_cells, grid_step))

    # Layer summary
    lines.extend(_format_layer_summary(result.instances, grid_step))

    return "\n".join(lines)


def _format_blocked_zones(
    blocked: np.ndarray, num_rows: int, num_cells: int, grid_step: int
) -> List[str]:
    """Format blocked zones info."""
    lines = ["", "Blocked zones per layer:"]

    for layer in range((num_rows + 1) // 2):
        row0 = layer * 2
        row1 = min(layer * 2 + 1, num_rows - 1)
        left_blocked, right_blocked = 0, 0

        for c in range(num_cells):
            if blocked[row0, c] or (row1 < num_rows and blocked[row1, c]):
                if c < num_cells // 2:
                    left_blocked = max(left_blocked, c + 1)
                else:
                    right_blocked = max(right_blocked, num_cells - c)

        if left_blocked > 0 or right_blocked > 0:
            lines.append(
                f"  L{layer}: left={left_blocked * grid_step}mm, right={right_blocked * grid_step}mm"
            )

    return lines


def _format_layer_summary(instances: Dict, grid_step: int) -> List[str]:
    """Format layer summary (Revit friendly)."""
    layer_map = defaultdict(list)

    for inst in instances.values():
        layer = inst["row"] // 2
        length_mm = (inst["end_cell"] - inst["start_cell"]) * grid_step
        layer_map[layer].append((inst["type_id"], length_mm))

    lines = ["", "Layer summary:"]

    for layer in sorted(layer_map.keys()):
        parts = []
        mono_sum = 0

        for type_id, length_mm in layer_map[layer]:
            if type_id == 0:
                mono_sum += length_mm
            else:
                parts.append(f"{type_id}({length_mm}mm)")

        if mono_sum > 0:
            parts.append(f"0({mono_sum}mm)")

        lines.append(f"  L{layer} | " + ", ".join(parts))

    return lines


def visualize_pipeline(
    result: PipelineResult, walls: List["WallInstance"], grid_step: int = 20
) -> str:
    """Visualize all walls in pipeline."""
    lines = ["=" * 70, "PIPELINE VISUALIZATION", "=" * 70]

    for wall in walls:
        wall_result = result.wall_results.get(wall.id)
        if wall_result:
            lines.extend(["", visualize_wall(wall_result, wall, grid_step), "-" * 70])

    return "\n".join(lines)
