# fbs_solver_beam.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


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
        beam_width: int = 32,
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

        # Sort FBS blocks by length exclude monolith
        self.fbs_blocks = sorted(
            [bt for bt in block_types if bt.id != 0],
            key=lambda bt: bt.length,
            reverse=True
        )
        self.monolith = next((bt for bt in block_types if bt.id == 0), None)
        self.min_fbs_cells = min(bt.length for bt in self.fbs_blocks) // grid_step

    def solve_wall(self):

        for row in range(self.num_rows):
            segments = self._get_free_segments(row)
            for start, end in segments:
                self._solve_segment_smart(row, start, end)

        return self.grid, self.instances

    # ============================================================
    # Segmeng solver
    # ============================================================

    def _solve_segment_smart(self, row: int, start: int, end: int):
        """
        Solve segment with rules from fbs_builder:
        1. First compute where FBS CAN fit (any FBS block can cover this cell)
        2. Place FBS blocks using beam search
        3. Monolith ONLY in cells where no FBS can fit
        """
        segment_len = end - start

        # If segment too small for any FBS, fill with monolith
        if segment_len < self.min_fbs_cells:
            self._fill_monolith(row, start, end)
            return

        fbs_can_fit = self._compute_fbs_coverage(row, start, end)

        must_mono = ~fbs_can_fit

        if not np.any(must_mono):
            self._solve_segment_beam(row, start, end)
            return
        
        self._solve_segment_mixed(row, start, end, fbs_can_fit)

    def _can_fill_without_mono(self, row: int, start: int, end: int) -> bool:
        """
        Check if segment can be filled 100% with FBS blocks (no monolith needed).
        Uses recursive check with memoization.
        """
        segment_len = end - start

        if segment_len == 0:
            return True

        if segment_len < self.min_fbs_cells:
            return False

        # Try each FBS block at start position
        for bt in self.fbs_blocks:
            cells = bt.length // self.grid_step
            h_rows = bt.height // self.row_height

            if cells > segment_len:
                continue
            if row + h_rows > self.num_rows:
                continue
            if not self._check_empty(row, start, cells, h_rows):
                continue
            if not self._check_support(row, start, cells):
                continue
            if not self._check_seam(row, start, cells, bt.height):
                continue

            # This block fits at start - can we fill the rest?
            remaining = segment_len - cells
            if remaining == 0:
                return True
            if remaining >= self.min_fbs_cells:
                # Recursively check remaining segment
                if self._can_fill_without_mono(row, start + cells, end):
                    return True

        return False

    def _compute_fbs_coverage(self, row: int, start: int, end: int) -> np.ndarray:
        """
        For each cell in [start, end), check if ANY FBS block placement
        can cover this cell (respecting all constraints).
        """
        segment_len = end - start
        can_fit = np.zeros(segment_len, dtype=bool)

        for bt in self.fbs_blocks:
            cells = bt.length // self.grid_step
            h_rows = bt.height // self.row_height

            if cells > segment_len:
                continue
            if row + h_rows > self.num_rows:
                continue

            # Try all possible start positions for this block
            for pos in range(start, end - cells + 1):
                if not self._check_empty(row, pos, cells, h_rows):
                    continue
                if not self._check_support(row, pos, cells):
                    continue
                if not self._check_seam(row, pos, cells, bt.height):
                    continue

                # This placement is valid - mark all covered cells
                local_start = pos - start
                local_end = local_start + cells
                can_fit[local_start:local_end] = True

        return can_fit

    def _solve_segment_mixed(self, row: int, start: int, end: int, fbs_can_fit: np.ndarray):
        """
        Solve segment with mixed FBS/mono zones.
        First place monolith in must-mono zones, then beam search for FBS.
        """
        # contiguous mono zones at edges (where we must use monolith)
        segment_len = end - start

        # Place monolith at left edge if needed
        mono_left = 0
        while mono_left < segment_len and not fbs_can_fit[mono_left]:
            mono_left += 1

        if mono_left > 0:
            self._fill_monolith(row, start, start + mono_left)

        # Place monolith at right edge if needed
        mono_right = 0
        while mono_right < segment_len - mono_left and not fbs_can_fit[segment_len - 1 - mono_right]:
            mono_right += 1

        if mono_right > 0:
            self._fill_monolith(row, end - mono_right, end)

        # Solve middle part with beam search
        new_start = start + mono_left
        new_end = end - mono_right

        if new_end > new_start:
            if new_end - new_start >= self.min_fbs_cells:
                self._solve_segment_beam(row, new_start, new_end)
            else:
                self._fill_monolith(row, new_start, new_end)

    def _fill_monolith(self, row: int, start: int, end: int):
        """Fill range with monolith."""
        for pos in range(start, end):
            # Skip if already filled
            if self.grid[row, pos] != 0:
                continue

            # Don't place monolith at absolute wall start (pos 0) if no blocks to left
            # and wall doesn't have blocked zone there
            is_left_edge = pos == 0
            is_right_edge = pos == self.num_cells - 1

            # Check if there are blocks adjacent (then mono is OK as filler)
            has_block_left = pos > 0 and self.grid[row, pos - 1] > 0
            has_block_right = pos < self.num_cells - 1 and self.grid[row, pos + 1] > 0

            has_blocked_at_pos = self.blocked[row, pos] == 1

            # Allow monolith if:
            # - Not at edge, OR
            # - At edge but has adjacent block (filler), OR
            # - At edge but has blocked zone (neighbor wall)
            if is_left_edge and not has_block_right and not has_blocked_at_pos:
                continue
            if is_right_edge and not has_block_left and not has_blocked_at_pos:
                continue

            self._place_block(row, pos, 1, 1, 0)

    # ============================================================
    # BEAM SEARCH (for FBS-only segments)
    # ============================================================

    def _solve_segment_beam(self, row: int, start: int, end: int):
        """Beam search for FBS blocks only (no monolith fallback)."""

        # Check if segment can be filled 100% with FBS (no monolith)
        can_fill_fully = self._can_fill_without_mono(row, start, end)

        # Adjust mono penalty based on whether full FBS fill is possible
        # If we CAN fill without mono, heavy penalty for using mono
        # If we CANNOT, normal penalty
        mono_penalty_multiplier = 10.0 if can_fill_fully else 1.0

        # Collect types already used in this wall (from previous rows)
        used_types_wall = set()
        for inst in self.instances.values():
            if inst["type_id"] != 0:  # Exclude monolith
                used_types_wall.add(inst["type_id"])

        initial_state = {
            "pos": start,
            "placements": [],
            "score": 0,
            "gaps": [], 
            "types_used": set(used_types_wall)
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

                for bt in self.fbs_blocks:
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

                    # Bonus for reusing same type, penalty for new type
                    type_bonus = 20.0 if bt.id in state["types_used"] else -15.0
                    new_types = state["types_used"] | {bt.id}

                    # Check if this creates unfillable gap (will need monolith)
                    remaining = end - (pos + cells)
                    if 0 < remaining < self.min_fbs_cells:
                        mono_penalty = remaining * mono_penalty_multiplier

                        gap_types = new_types | {0}
                        gap_type_penalty = 0 if 0 in state["types_used"] else -15.0
                        new_state = {
                            "pos": end,  # Skip to end gap will be filled with mono
                            "placements": state["placements"] + [(pos, cells, h_rows, bt.id)],
                            "score": state["score"] + self._score_block(bt, cells) + type_bonus - mono_penalty + gap_type_penalty,
                            "gaps": state["gaps"] + [(pos + cells, end)],
                            "types_used": gap_types
                        }
                        new_beam.append(new_state)

                    placed_any = True

                    new_state = {
                        "pos": pos + cells,
                        "placements": state["placements"] + [(pos, cells, h_rows, bt.id)],
                        "score": state["score"] + self._score_block(bt, cells) + type_bonus,
                        "gaps": state["gaps"],
                        "types_used": new_types
                    }
                    new_beam.append(new_state)

                # If no FBS fits, try skipping one cell (will be mono later)
                if not placed_any:
                    skip_types = state["types_used"] | {0}
                    skip_type_penalty = 0 if 0 in state["types_used"] else -15.0
                    new_state = {
                        "pos": pos + 1,
                        "placements": state["placements"],
                        "score": state["score"] - 10 * mono_penalty_multiplier + skip_type_penalty,
                        "gaps": state["gaps"] + [(pos, pos + 1)],
                        "types_used": skip_types
                    }
                    new_beam.append(new_state)

            if finished:
                break

            # Keep top-K states
            new_beam.sort(key=lambda s: s["score"], reverse=True)
            beam = new_beam[:self.beam_width]

        # Pick best finished state
        best = max(beam, key=lambda s: s["score"])

        # Place FBS blocks
        for pos, cells, h_rows, type_id in best["placements"]:
            self._place_block(row, pos, cells, h_rows, type_id)

        # Fill gaps with monolith
        for gap_start, gap_end in best["gaps"]:
            self._fill_monolith(row, gap_start, gap_end)

    # ============================================================
    # Score
    # ============================================================

    def _score_block(self, bt: BlockType, cells: int) -> float:
        """
        Score prioritizes:
        1. Fewer blocks (larger blocks preferred)
        2. Less monolith (but acceptable if reduces block count)

        Trade-off example:
        - 3x ФБС-9 (900mm) = 3 blocks, 0 mono
        - 2x ФБС-12 (1200mm) + 300mm mono = 2 blocks, 15 cells mono

        We prefer 2 large blocks + small mono over 3 small blocks.
        """
        if bt.id == 0:
            # Monolith: small penalty per cell
            return -1.0 * cells

        size_bonus = cells * 2.0 
        block_penalty = 30  # Each block costs -30

        return size_bonus - block_penalty

    # ============================================================
    # Checks
    # ============================================================

    def _check_empty(self, row: int, start: int, cells: int, h_rows: int) -> bool:
        return np.all(self.grid[row:row + h_rows, start:start + cells] == 0)

    def _check_support(self, row: int, start: int, cells: int) -> bool:
        if row == 0:
            return True
        below = self.grid[row - 1, start:start + cells]
        return np.any(below != 0)

    def _check_seam(self, row: int, start: int, cells: int, block_height: int) -> bool:
        if row == 0:
            return True

        min_shift_mm = block_height * self.min_seam_shift_ratio
        min_shift_cells = int(min_shift_mm // self.grid_step)

        seams_below = self._find_seams(row - 1)

        left = start
        right = start + cells

        for seam in seams_below:
            # Seams must NOT coincide and must be at least min_shift apart
            dist_left = abs(seam - left)
            dist_right = abs(seam - right)

            # Exact match (dist=0) is violation, close match (dist < min_shift) is also violation
            if dist_left < min_shift_cells:
                return False
            if dist_right < min_shift_cells:
                return False

        return True

    def _find_seams(self, row: int) -> List[int]:
        """Find block boundaries, ignoring monolith cells."""
        seams = []
        for i in range(self.num_cells - 1):
            id_left = self.grid[row, i]
            id_right = self.grid[row, i + 1]

            # Skip if either is empty or blocked
            if id_left <= 0 or id_right <= 0:
                continue

            # Skip if same block
            if id_left == id_right:
                continue

            # Check if both are monolith (type_id = 0) - ignore boundaries between monolith
            type_left = self.instances.get(id_left, {}).get("type_id", -1)
            type_right = self.instances.get(id_right, {}).get("type_id", -1)

            if type_left == 0 and type_right == 0:
                continue  # Both monolith - not a real seam

            seams.append(i + 1)

        return seams

    # ============================================================
    # P;ace
    # ============================================================

    def _place_block(self, row: int, start: int, cells: int, h_rows: int, type_id: int):
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
    # Openings
    # ============================================================

    def _apply_openings(self, openings: List[Opening]):
        for op in openings:
            x0 = max(0, (op.center_x - op.width // 2) // self.grid_step)
            x1 = min(self.num_cells, (op.center_x + op.width // 2) // self.grid_step)
            y0 = max(0, (op.center_y - op.height // 2) // self.row_height)
            y1 = min(self.num_rows,
                     (op.center_y + op.height // 2 + self.row_height - 1) // self.row_height)

            self.blocked[y0:y1, x0:x1] = 1
            self.grid[y0:y1, x0:x1] = -1

    def _get_free_segments(self, row: int) -> List[Tuple[int, int]]:
        segments = []
        in_seg = False
        seg_start = 0

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
