from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np

from src.builder.structures import WallInstance, GRID_STEP


class ContextBuilder:
    """
    Builds constraint grids for RL agent.

    Bonding types:
        0 = block layers 0, 2, 4... (even blocks)
        1 = block layers 1, 3, 5... (odd blocks)

    The bonding pattern creates chess-like interlocking between adjacent walls.
    """

    def __init__(self, grid_step=20):
        self.grid_step = grid_step

    def build_grid(
        self,
        walls: list,
        current_idx: int,
        num_rows: int,
        num_cells: int,
        bonding_left: Optional[int] = None,
        bonding_right: Optional[int] = None,
        context_data: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Build constraint grid for a wall. Used in FBSBuilderEnv

        Args:
            walls: List of all walls
            current_idx: Index of current wall being built
            num_rows: Height in 300mm rows
            num_cells: Width in grid cells
            bonding_left: Bonding type for left joint (0 or 1), None = auto-detect
            bonding_right: Bonding type for right joint (0 or 1), None = auto-detect
            context_data: Optional dict with explicit bonding/neighbor info from optimizer

        Returns:
            Grid where 1 = blocked, 0 = free
        """
        # Check if optimizer provided explicit bonding info
        if context_data:
            if "bonding_left" in context_data:
                bonding_left = context_data["bonding_left"]
            if "bonding_right" in context_data:
                bonding_right = context_data["bonding_right"]
            if "left_wall" in context_data:
                completed_wall = context_data["left_wall"]
            else:
                completed_wall = walls[current_idx - 1] if current_idx > 0 else None
            if "right_wall" in context_data:
                next_wall = context_data["right_wall"]
            else:
                next_wall = walls[current_idx + 1] if current_idx + 1 < len(walls) else None
        else:
            completed_wall = walls[current_idx - 1] if current_idx > 0 else None
            next_wall = walls[current_idx + 1] if current_idx + 1 < len(walls) else None

        grid = np.zeros((num_rows, num_cells), dtype=np.uint8)

        # Auto-detect bonding if not specified (backward compatibility)
        if bonding_left is None and completed_wall is not None:
            bonding_left = 0  # Default: block even layers on left
        if bonding_right is None and next_wall is not None:
            bonding_right = 1  # Default: block odd layers on right

        grid = self._apply_end_restrictions(
            grid,
            completed_wall,
            next_wall,
            bonding_left,
            bonding_right
        )

        return grid

    def build_grid_with_bonding(
        self,
        wall: WallInstance,
        left_wall: Optional[WallInstance],
        right_wall: Optional[WallInstance],
        bonding_left: Optional[int],
        bonding_right: Optional[int],
    ) -> np.ndarray:
        """
        Build constraint grid with explicit bonding types.
        Simplified interface for BondingOptimizer.

        Args:
            wall: Current wall to build
            left_wall: Adjacent wall on left (for thickness), None if no neighbor
            right_wall: Adjacent wall on right (for thickness), None if no neighbor
            bonding_left: 0 or 1, None if no left neighbor
            bonding_right: 0 or 1, None if no right neighbor
        """
        num_rows = wall.num_rows
        num_cells = wall.num_cells

        grid = np.zeros((num_rows, num_cells), dtype=np.uint8)

        grid = self._apply_end_restrictions(
            grid,
            left_wall,
            right_wall,
            bonding_left,
            bonding_right
        )

        return grid

    def _apply_end_restrictions(
            self,
            grid: np.ndarray,
            completed_wall: Optional[WallInstance],
            next_wall: Optional[WallInstance],
            bonding_left: Optional[int] = None,
            bonding_right: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply chess-pattern restrictions at wall joints.

        Args:
            grid: Constraint grid to modify
            completed_wall: Left neighbor (for thickness)
            next_wall: Right neighbor (for thickness)
            bonding_left: 0 = block even layers, 1 = block odd layers
            bonding_right: 0 = block even layers, 1 = block odd layers
        """

        # Left end (from completed wall)
        if completed_wall and bonding_left is not None:
            r_width_comp = completed_wall.weight
            cells_comp = r_width_comp // self.grid_step

            for layer in range(grid.shape[0]):
                block = layer // 2
                # bonding_left=0: block layers where block%2==0 (even)
                # bonding_left=1: block layers where block%2==1 (odd)
                if block % 2 == bonding_left:
                    grid[layer, :cells_comp] = 1

        # Right end (towards the next wall)
        if next_wall and bonding_right is not None:
            r_width_next = next_wall.weight
            cells_next = r_width_next // self.grid_step

            for layer in range(grid.shape[0]):
                block = layer // 2
                if block % 2 == bonding_right:
                    grid[layer, -cells_next:] = 1

        return grid