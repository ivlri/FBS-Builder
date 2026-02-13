"""
Solver pipeline for multi-wall processing.

Unlike BondingOptimizer (RL-based), this uses deterministic beam search solver.
No bonding optimization needed - solver is deterministic, so we just propagate
constraints between walls.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.solver.solver import FBSSolver, BlockType as SolverBlockType, WallInstance as SolverWall, Opening
from src.builder.structures import WallInstance, GRID_STEP, BLOCK_TYPES, BlockType


@dataclass
class SolverResult:
    """Result for a single wall."""
    wall_id: int
    grid: np.ndarray
    instances: Dict[int, Dict]
    stats: Dict
    blocked_mask: Optional[np.ndarray] = None  # For debugging


@dataclass
class PipelineResult:
    """Result for entire wall chain."""
    wall_results: Dict[int, SolverResult]
    total_stats: Dict


class SolverPipeline:
    """
    Process chain of walls with solver, propagating constraints.

    Chess-pattern bonding:
    - bonding=0: block layers 0, 2, 4 (even 600mm layers)
    - bonding=1: block layers 1, 3, 5 (odd 600mm layers)

    Flow:
    1. Process walls sequentially (left to right)
    2. For each wall:
       - Apply chess-pattern bonding at joints
       - Apply occupied cells from neighbors (prevent overlap)
       - Run solver with constraints
       - Extract right edge for next wall
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

        # Convert BLOCK_TYPES to solver format
        self.block_types = [
            SolverBlockType(id=bt.id, length=bt.length, height=bt.height)
            for bt in BLOCK_TYPES
        ]

    def solve_chain(
        self,
        walls: List[WallInstance],
        initial_bonding: int = 0,
    ) -> PipelineResult:
        """
        Solve chain of connected walls with chess-pattern bonding.

        Args:
            walls: Ordered list of walls (left to right)
            initial_bonding: Starting bonding pattern (0 or 1)
                0 = first wall blocks even layers (0,2,4) on right
                1 = first wall blocks odd layers (1,3,5) on right

        Returns:
            PipelineResult with all wall layouts
        """
        if not walls:
            return PipelineResult({}, {"total_fbs": 0, "total_mono": 0})

        results: Dict[int, SolverResult] = {}
        left_occupied: Optional[np.ndarray] = None
        current_bonding = initial_bonding

        total_fbs = 0
        total_mono = 0

        for i, wall in enumerate(walls):
            right_wall = walls[i + 1] if i < len(walls) - 1 else None
            left_wall = walls[i - 1] if i > 0 else None

            num_rows = wall.height // self.row_height

            # Build blocked mask with chess-pattern
            blocked = np.zeros((num_rows, wall.num_cells), dtype=bool)

            # Chess-pattern logic:
            # bonding=0 means: BLOCK even layers (0,2,4), so wall PLACES on odd layers (1,3,5)
            # bonding=1 means: BLOCK odd layers (1,3,5), so wall PLACES on even layers (0,2,4)
            #
            # For interlocking:
            # - Wall N blocks layer X on right → Wall N places on layer (1-X) on right
            # - Wall N+1 must block layer (1-X) on left (where Wall N placed blocks)
            # - So Wall N+1 bonding_left = 1 - bonding_right_of_wall_N

            # Apply chess-pattern bonding on RIGHT edge (this wall blocks some layers)
            bonding_right = current_bonding  # This wall's right side pattern
            if right_wall is not None:
                width_cells = right_wall.weight // self.grid_step

                for row in range(num_rows):
                    layer = row // 2
                    if layer % 2 == bonding_right:
                        blocked[row, -width_cells:] = True

            # Apply chess-pattern bonding on LEFT edge (block where neighbor PLACED blocks)
            # Neighbor used bonding_right = prev_bonding, so it PLACED on layers != prev_bonding
            # We must BLOCK those layers = bonding_left = 1 - prev_bonding
            if left_wall is not None and i > 0:
                width_cells = left_wall.weight // self.grid_step
                # Previous wall's bonding was (current_bonding inverted once)
                # After prev wall: current_bonding = 1 - prev_bonding
                # So prev_bonding = 1 - current_bonding
                # Neighbor placed on layers != prev_bonding, i.e. == current_bonding
                # We block those: bonding_left = current_bonding
                bonding_left = current_bonding

                for row in range(num_rows):
                    layer = row // 2
                    if layer % 2 == bonding_left:
                        blocked[row, :width_cells] = True

            # Also block occupied cells from left neighbor (prevent overlap)
            if left_occupied is not None:
                min_rows = min(num_rows, left_occupied.shape[0])
                min_cols = min(wall.num_cells, left_occupied.shape[1])
                for r in range(min_rows):
                    for c in range(min_cols):
                        if left_occupied[r, c]:
                            blocked[r, c] = True

            # Convert openings to solver format
            solver_openings = self._convert_openings(wall)

            # Create solver
            solver_wall = SolverWall(length=wall.length, height=wall.height)
            solver = FBSSolver(
                wall=solver_wall,
                block_types=self.block_types,
                openings=solver_openings,
                grid_step=self.grid_step,
                row_height=self.row_height,
                beam_width=self.beam_width,
            )

            # Apply blocked zones
            self._apply_blocked_mask(solver, blocked)

            # Solve
            grid, instances = solver.solve_wall()

            # Compute stats
            stats = self._compute_stats(instances)
            total_fbs += stats["fbs_count"]
            total_mono += stats["monolith_cells"]

            # Save result
            results[wall.id] = SolverResult(
                wall_id=wall.id,
                grid=grid,
                instances=instances,
                stats=stats,
                blocked_mask=blocked.copy(),
            )

            # Extract right edge for next wall
            if right_wall is not None:
                width_cells = right_wall.weight // self.grid_step
                left_occupied = self._extract_edge(grid, "right", width_cells)
                # Alternate bonding for next wall
                # This wall used bonding_right = current_bonding
                # Next wall's bonding_left will use current_bonding (after inversion)
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

    def _apply_blocked_mask(self, solver: FBSSolver, blocked: np.ndarray) -> None:
        """Apply blocked mask to solver."""
        rows, cols = blocked.shape
        rows = min(rows, solver.num_rows)
        cols = min(cols, solver.num_cells)

        for r in range(rows):
            for c in range(cols):
                if blocked[r, c]:
                    solver.blocked[r, c] = 1
                    solver.grid[r, c] = -1

    def _convert_openings(self, wall: WallInstance) -> Optional[List[Opening]]:
        """Convert WallInstance openings to solver Opening format."""
        if not hasattr(wall, "openings") or not wall.openings:
            return None

        solver_openings = []
        for op in wall.openings:
            # Both use same format: center_x, center_y, width, height
            solver_openings.append(Opening(
                center_x=op.center_x,
                center_y=op.center_y,
                width=op.width,
                height=op.height,
            ))

        return solver_openings


    def _extract_edge(
        self,
        grid: np.ndarray,
        side: str,
        width_cells: int,
    ) -> np.ndarray:
        """Extract occupied cells from edge of grid."""
        if side == "right":
            edge = grid[:, -width_cells:]
        else:  # left
            edge = grid[:, :width_cells]

        # Binary: occupied (>0) or not
        return edge > 0

    def _compute_stats(self, instances: Dict) -> Dict:
        """Compute statistics for wall layout."""
        fbs_count = 0
        monolith_cells = 0
        fbs_cells = 0

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


def visualize_wall(result: SolverResult, wall: 'WallInstance', grid_step: int = 20) -> str:
    """
    Create ASCII visualization of wall with blocked zones and blocks.

    Legend:
        X = blocked (chess-pattern or occupied by neighbor)
        . = empty
        0 = monolith
        2-7 = FBS block type
    """
    grid = result.grid
    blocked = result.blocked_mask
    num_rows, num_cells = grid.shape

    # Create type grid from instances
    type_grid = np.zeros_like(grid)
    for inst in result.instances.values():
        type_id = inst["type_id"]
        row = inst["row"]
        h_rows = inst.get("h_rows", 1)
        start = inst["start_cell"]
        end = inst["end_cell"]
        type_grid[row:row+h_rows, start:end] = type_id if type_id > 0 else -2  # -2 for monolith display

    lines = []
    lines.append(f"Wall {result.wall_id}: {wall.length}mm x {wall.height}mm")
    lines.append(f"Grid: {num_rows} rows x {num_cells} cells ({grid_step}mm/cell)")
    lines.append("")

    # Compress visualization: show every 10th cell or summarize
    step = max(1, num_cells // 60)  # Max 60 chars wide

    # Header with positions
    header = "     "
    for c in range(0, num_cells, step * 10):
        header += f"{c:<10}"
    lines.append(header)

    # Rows from top to bottom
    for row in range(num_rows - 1, -1, -1):
        layer = row // 2
        row_str = f"R{row:02d} "

        for c in range(0, num_cells, step):
            if blocked is not None and blocked[row, c]:
                row_str += "X"
            elif grid[row, c] == -1:
                row_str += "X"
            elif grid[row, c] == 0:
                row_str += "."
            else:
                t = type_grid[row, c]
                if t == -2:
                    row_str += "M"  # Monolith
                elif t == 0:
                    row_str += "M"
                else:
                    row_str += str(t % 10)

        row_str += f" | L{layer}"
        lines.append(row_str)

    # Blocked zones summary
    if blocked is not None:
        lines.append("")
        lines.append("Blocked zones per layer:")
        for layer in range((num_rows + 1) // 2):
            row0 = layer * 2
            row1 = min(layer * 2 + 1, num_rows - 1)

            left_blocked = 0
            right_blocked = 0

            for c in range(num_cells):
                if blocked[row0, c] or (row1 < num_rows and blocked[row1, c]):
                    if c < num_cells // 2:
                        left_blocked = max(left_blocked, c + 1)
                    else:
                        right_blocked = max(right_blocked, num_cells - c)

            if left_blocked > 0 or right_blocked > 0:
                lines.append(f"  L{layer}: left={left_blocked * grid_step}mm, right={right_blocked * grid_step}mm")

    return "\n".join(lines)


def visualize_pipeline(result: PipelineResult, walls: List['WallInstance'], grid_step: int = 20) -> str:
    """Visualize all walls in pipeline."""
    lines = []
    lines.append("=" * 70)
    lines.append("PIPELINE VISUALIZATION")
    lines.append("=" * 70)

    for wall in walls:
        wall_result = result.wall_results.get(wall.id)
        if wall_result:
            lines.append("")
            lines.append(visualize_wall(wall_result, wall, grid_step))
            lines.append("-" * 70)

    return "\n".join(lines)
