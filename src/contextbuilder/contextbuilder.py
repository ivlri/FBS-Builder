from typing import List, Tuple, Dict, Any, Set, Optional
from src.builder.structures import WallInstance, GRID_STEP
import numpy as np

class ContextBuilder:
    def __init__(self, grid_step=20):
        self.grid_step = grid_step

    def build_grid(
        self,
        walls: list,
        current_idx: int,
        num_rows: int,
        num_cells: int,
    ) -> np.ndarray:

        completed_wall = walls[current_idx - 1] if current_idx > 0 else None
        next_wall = walls[current_idx + 1] if current_idx + 1 < len(walls) else None

        grid = np.zeros((num_rows, num_cells), dtype=np.uint8)

        grid = self._apply_end_restrictions(
            grid,
            completed_wall,
            next_wall
        )

        return grid
    
    def _apply_end_restrictions(
            self, 
            grid, 
            completed_wall: Optional[WallInstance], 
            next_wall: Optional[WallInstance]
    ):
        
        # Left end (from completed wall)
        if completed_wall:
            r_width_comp = completed_wall.weight
            cells_comp = r_width_comp // self.grid_step
            
            for layer in range(grid.shape[0]):
                block = layer // 2
                if block % 2 != 1:  # every second layer is blocked 
                    grid[layer,:cells_comp] = 1
        
        # Right end (towards the next wall)
        if next_wall:
            r_width_next = next_wall.weight
            cells_next = r_width_next // self.grid_step

            for layer in range(grid.shape[0]):
                block = layer // 2
                if block % 2 == 1:  # every second layer is blocked 
                    grid[layer, -cells_next:] = 1
        
        return grid