# fbs_solver_beam.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy


@dataclass
class BlockType:
    id: int
    length: int
    height: int


@dataclass
class WallInstance:
    length: int
    height: int


@dataclass
class Opening:
    center_x: int
    center_y: int
    width: int
    height: int


class FBSSolver:

    def __init__(
        self,
        wall: WallInstance,
        block_types: List[BlockType],
        openings: Optional[List[Opening]] = None,
        grid_step: int = 20,
        row_height: int = 300,
        min_seam_shift_ratio: float = 0.4,
        beam_width: int = 8,
    ):

        self.wall = wall
        self.block_types = block_types
        self.grid_step = grid_step
        self.row_height = row_height
        self.min_seam_shift_ratio = min_seam_shift_ratio
        self.beam_width = beam_width

        self.num_cells = wall.length // grid_step
        self.num_rows = wall.height // row_height

        self.grid = np.zeros((self.num_rows, self.num_cells), dtype=np.int32)
        self.blocked = np.zeros_like(self.grid)

        if openings:
            self._apply_openings(openings)

        self.instance_counter = 1
        self.instances: Dict[int, Dict] = {}

        self.block_types_sorted = sorted(
            block_types,
            key=lambda bt: bt.length,
            reverse=True
        )

    # ============================================================
    # MAIN
    # ============================================================

    def solve_wall(self):

        for row in range(self.num_rows):
            segments = self._get_free_segments(row)
            for start, end in segments:
                self._solve_segment_beam(row, start, end)

        return self.grid, self.instances

    # ============================================================
    # SEGMENT BEAM SEARCH
    # ============================================================

    def _solve_segment_beam(self, row, start, end):

        initial_state = {
            "pos": start,
            "placements": [],
            "score": 0
        }

        beam = [initial_state]

        while True:

            new_beam = []
            finished = True

            for state in beam:

                pos = state["pos"]

                if pos >= end:
                    new_beam.append(state)
                    continue

                finished = False

                placed_any = False

                for bt in self.block_types_sorted:

                    cells = bt.length // self.grid_step
                    h_rows = bt.height // self.row_height

                    if pos + cells > end:
                        continue
                    if row + h_rows > self.num_rows:
                        continue
                    if not self._check_empty(row, pos, cells, h_rows):
                        continue
                    if not self._check_support(row, pos, cells):
                        continue
                    if not self._check_seam(row, pos, cells, bt.height):
                        continue

                    placed_any = True

                    new_state = {
                        "pos": pos + cells,
                        "placements": state["placements"] + [(pos, cells, h_rows, bt.id)],
                        "score": state["score"] + self._score_block(bt, cells)
                    }

                    new_beam.append(new_state)

                # fallback monolith
                if not placed_any:
                    new_state = {
                        "pos": pos + 1,
                        "placements": state["placements"] + [(pos, 1, 1, 0)],
                        "score": state["score"] - 2
                    }
                    new_beam.append(new_state)

            if finished:
                break

            # keep top-K states
            new_beam.sort(key=lambda s: s["score"], reverse=True)
            beam = new_beam[:self.beam_width]

        # pick best finished
        best = max(beam, key=lambda s: s["score"])

        for pos, cells, h_rows, type_id in best["placements"]:
            self._place_block(row, pos, cells, h_rows, type_id)

    # ============================================================
    # SCORING
    # ============================================================

    def _score_block(self, bt, cells):

        if bt.id == 0:
            return -5

        size_bonus = cells
        return 10 + size_bonus

    # ============================================================
    # CHECKS
    # ============================================================

    def _check_empty(self, row, start, cells, h_rows):
        return np.all(self.grid[row:row + h_rows, start:start + cells] == 0)

    def _check_support(self, row, start, cells):
        if row == 0:
            return True
        below = self.grid[row - 1, start:start + cells]
        return np.any(below != 0)

    def _check_seam(self, row, start, cells, block_height):

        if row == 0:
            return True

        min_shift_mm = block_height * self.min_seam_shift_ratio
        min_shift_cells = int(min_shift_mm // self.grid_step)

        seams_below = self._find_seams(row - 1)

        left = start
        right = start + cells

        for seam in seams_below:
            if abs(seam - left) < min_shift_cells:
                return False
            if abs(seam - right) < min_shift_cells:
                return False

        return True

    def _find_seams(self, row):

        seams = []
        for i in range(self.num_cells - 1):
            if (
                self.grid[row, i] > 0
                and self.grid[row, i + 1] > 0
                and self.grid[row, i] != self.grid[row, i + 1]
            ):
                seams.append(i + 1)

        return seams

    # ============================================================
    # PLACE
    # ============================================================

    def _place_block(self, row, start, cells, h_rows, type_id):

        inst_id = self.instance_counter
        self.instance_counter += 1

        self.instances[inst_id] = {
            "row": row,
            "start_cell": start,
            "end_cell": start + cells,
            "h_rows": h_rows,
            "type_id": type_id,
        }

        self.grid[row:row + h_rows, start:start + cells] = inst_id

    # ============================================================
    # OPENINGS / SEGMENTS
    # ============================================================

    def _apply_openings(self, openings):

        for op in openings:
            x0 = max(0, (op.center_x - op.width // 2) // self.grid_step)
            x1 = min(self.num_cells, (op.center_x + op.width // 2) // self.grid_step)
            y0 = max(0, (op.center_y - op.height // 2) // self.row_height)
            y1 = min(self.num_rows,
                     (op.center_y + op.height // 2 + self.row_height - 1) // self.row_height)

            self.blocked[y0:y1, x0:x1] = 1
            self.grid[y0:y1, x0:x1] = -1

    def _get_free_segments(self, row):

        segments = []
        in_seg = False

        for c in range(self.num_cells):

            if self.blocked[row, c] == 0 and self.grid[row, c] == 0:
                if not in_seg:
                    seg_start = c
                    in_seg = True
            else:
                if in_seg:
                    segments.append((seg_start, c))
                    in_seg = False

        if in_seg:
            segments.append((seg_start, self.num_cells))

        return segments
