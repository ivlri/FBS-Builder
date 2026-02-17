from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


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
        left_overhang_mm: int = 0,
        right_overhang_mm: int = 0,
    ):
        self.wall = wall
        self.block_types = block_types
        self.grid_step = grid_step
        self.row_height = row_height
        self.min_seam_shift_ratio = min_seam_shift_ratio
        self.beam_width = beam_width

        self.left_overhang_cells = left_overhang_mm // grid_step
        self.right_overhang_cells = right_overhang_mm // grid_step
        self.left_overhang_mm = left_overhang_mm
        self.right_overhang_mm = right_overhang_mm

        self.num_cells = wall.length // grid_step
        self.num_rows = wall.height // row_height

        self.grid = np.zeros((self.num_rows, self.num_cells), dtype=np.int32)
        self.blocked = np.zeros_like(self.grid)
        self.openings = openings or []

        if openings:
            self._apply_openings(openings)

        self.instance_counter = 1
        self.instances: Dict[int, Dict] = {}

        all_fbs = [bt for bt in block_types if bt.id != 0]
        self.fbs_600 = sorted(
            [bt for bt in all_fbs if bt.height == 600],
            key=lambda bt: bt.length,
            reverse=True,
        )
        self.fbs_300 = sorted(
            [bt for bt in all_fbs if bt.height == 300],
            key=lambda bt: bt.length,
            reverse=True,
        )
        self.fbs_blocks = self.fbs_600

        self.monolith = next((bt for bt in block_types if bt.id == 0), None)
        self.min_fbs_cells = min(bt.length for bt in all_fbs) // grid_step

    def _get_allowed_blocks(self, row: int) -> List[BlockType]:
        """Get allowed FBS blocks. 300mm blocks only near openings (±1 row)."""
        if not self.openings:
            return self.fbs_600

        for op in self.openings:
            y0 = (op.center_y - op.height // 2) // self.row_height
            y1 = (op.center_y + op.height // 2 + self.row_height - 1) // self.row_height

            if y0 - 1 <= row <= y1:
                return self.fbs_600 + self.fbs_300

        return self.fbs_600

    def solve_wall(self):
        for row in range(self.num_rows):
            segments = self._get_free_segments(row)
            for start, end in segments:
                self._solve_segment_smart(row, start, end)

        return self.grid, self.instances

    # ============================================================
    # Segment solver
    # ============================================================

    def _solve_segment_smart(self, row: int, start: int, end: int):
        """
        Solve segment with rules:
        1. Compute where FBS can fit
        2. Place FBS blocks using beam search
        3. Monolith ONLY where no FBS can fit
        """
        segment_len = end - start

        if segment_len < self.min_fbs_cells:
            self._fill_monolith(row, start, end)
            return

        fbs_can_fit = self._compute_fbs_coverage(row, start, end)

        if np.all(fbs_can_fit):
            self._solve_segment_beam(row, start, end)
        else:
            self._solve_segment_mixed(row, start, end, fbs_can_fit)

    def _compute_fbs_coverage(self, row: int, start: int, end: int) -> np.ndarray:
        """Check which cells can be covered by ANY valid FBS placement."""
        segment_len = end - start
        can_fit = np.zeros(segment_len, dtype=bool)

        for bt in self._get_allowed_blocks(row):
            cells = bt.length // self.grid_step
            h_rows = bt.height // self.row_height

            if cells > segment_len or row + h_rows > self.num_rows:
                continue

            for pos in range(start, end - cells + 1):
                if self._is_valid_placement(row, pos, cells, h_rows, bt.height):
                    local_start = pos - start
                    can_fit[local_start : local_start + cells] = True

        return can_fit

    def _solve_segment_mixed(
        self, row: int, start: int, end: int, fbs_can_fit: np.ndarray
    ):
        """Solve segment with mixed FBS/mono zones."""
        segment_len = end - start

        mono_left = 0
        while mono_left < segment_len and not fbs_can_fit[mono_left]:
            mono_left += 1

        if mono_left > 0:
            self._fill_monolith(row, start, start + mono_left)

        mono_right = 0
        while (
            mono_right < segment_len - mono_left
            and not fbs_can_fit[segment_len - 1 - mono_right]
        ):
            mono_right += 1

        if mono_right > 0:
            self._fill_monolith(row, end - mono_right, end)

        new_start = start + mono_left
        new_end = end - mono_right

        if new_end > new_start:
            if new_end - new_start >= self.min_fbs_cells:
                self._solve_segment_beam(
                    row,
                    new_start,
                    new_end,
                    has_mono_left=mono_left > 0,
                    has_mono_right=mono_right > 0,
                )
            else:
                self._fill_monolith(row, new_start, new_end)

    def _fill_monolith(self, row: int, start: int, end: int):
        """Fill range with monolith, respecting edge rules."""
        for pos in range(start, end):
            if self.grid[row, pos] != 0:
                continue

            if not self._can_place_mono_at(row, pos):
                continue

            self._place_block(row, pos, 1, 1, 0)

    def _can_place_mono_at(self, row: int, pos: int) -> bool:
        """Check if monolith can be placed at position (edge rules)."""
        is_left_edge = pos == 0
        is_right_edge = pos == self.num_cells - 1

        if not is_left_edge and not is_right_edge:
            return True

        has_block_left = pos > 0 and self.grid[row, pos - 1] > 0
        has_block_right = pos < self.num_cells - 1 and self.grid[row, pos + 1] > 0
        if is_left_edge:
            return has_block_right or self.blocked[row, pos] == 1
        if is_right_edge:
            return has_block_left or self.blocked[row, pos] == 1

        return True

    # ============================================================
    # Beam search
    # ============================================================

    def _solve_segment_beam(
        self,
        row: int,
        start: int,
        end: int,
        has_mono_left: bool = False,
        has_mono_right: bool = False,
    ):
        """Beam search for FBS blocks only."""
        used_types_wall = {
            inst["type_id"] for inst in self.instances.values() if inst["type_id"] != 0
        }

        initial_states = self._build_initial_states(
            row,
            start,
            end,
            used_types_wall,
            has_mono_right,
        )
        beam = initial_states

        while not all(s["pos"] >= end for s in beam):
            beam = self._expand_beam(beam, row, start, end)
            if not beam:
                self._fill_monolith(row, start, end)
                return

        best = max(beam, key=lambda s: s["score"])
        self._apply_solution(row, best)

    def _build_initial_states(
        self,
        row: int,
        start: int,
        end: int,
        used_types: set,
        has_mono_right: bool = False,
    ) -> List[Dict]:
        """Build initial beam states, including left-gap option."""
        states = [
            {
                "pos": start,
                "placements": [],
                "score": 0,
                "gaps": [],
                "types_used": set(used_types),
            }
        ]

        # Skip left-gap when mono already placed on right (consolidate)
        if has_mono_right:
            return states

        # Try starting with gap if left side has blocked neighbors
        if self._has_blocked_near(start, "left"):
            base_state = {
                "placements": [],
                "gaps": [],
                "types_used": set(used_types),
            }
            for gap_size in range(1, self.min_fbs_cells):
                gap_pos = start + gap_size
                if gap_pos >= end:
                    break
                if self._can_fill_gap_with_mono(row, start, gap_pos):
                    gap_score = self._compute_gap_score(
                        base_state,
                        gap_size,
                        gap_start=start,
                        gap_end=gap_pos,
                    )
                    states.append(
                        {
                            "pos": gap_pos,
                            "placements": [],
                            "score": gap_score,
                            "gaps": [(start, gap_pos)],
                            "types_used": set(used_types) | {0},
                        }
                    )

        return states

    def _has_blocked_near(self, pos: int, side: str) -> bool:
        """Check if any row has blocked cells near position."""
        for r in range(self.num_rows):
            if side == "left":
                if pos > 0 and self.blocked[r, pos - 1] == 1:
                    return True
                if self.blocked[r, pos] == 1:
                    return True
            else:
                if pos < self.num_cells and self.blocked[r, pos] == 1:
                    return True
                if pos < self.num_cells - 1 and self.blocked[r, pos + 1] == 1:
                    return True
        return False

    def _expand_beam(
        self, beam: List[Dict], row: int, start: int, end: int
    ) -> List[Dict]:
        """Expand beam by one step."""
        new_beam = []

        for state in beam:
            pos = state["pos"]

            if pos >= end:
                new_beam.append(state)
                continue

            placed = self._try_place_fbs(state, row, pos, end)
            new_beam.extend(placed)

            if not placed:
                skipped = self._try_skip_cell(state, row, pos)
                if skipped:
                    new_beam.append(skipped)

        new_beam.sort(key=lambda s: s["score"], reverse=True)
        return new_beam[: self.beam_width]

    def _try_place_fbs(self, state: Dict, row: int, pos: int, end: int) -> List[Dict]:
        """Try placing all possible FBS blocks at position."""
        new_states = []

        for bt in self._get_allowed_blocks(row):
            cells = bt.length // self.grid_step
            h_rows = bt.height // self.row_height

            if pos + cells > end or row + h_rows > self.num_rows:
                continue

            if not self._is_valid_placement(row, pos, cells, h_rows, bt.height):
                continue

            new_state = self._create_placement_state(
                state,
                bt,
                pos,
                cells,
                h_rows,
                end,
                row,
            )
            if new_state:
                new_states.append(new_state)

        return new_states

    def _create_placement_state(
        self,
        state: Dict,
        bt: BlockType,
        pos: int,
        cells: int,
        h_rows: int,
        end: int,
        row: int = 0,
    ) -> Optional[Dict]:
        """Create new state for FBS placement."""
        remaining = end - (pos + cells)

        if 0 < remaining < self.min_fbs_cells:
            if not self._can_fill_gap_with_mono(row, pos + cells, end):
                return None

            gap_start = pos + cells
            gap_score = self._compute_gap_score(
                state,
                remaining,
                gap_start=gap_start,
                gap_end=end,
            )
            return {
                "pos": end,
                "placements": state["placements"] + [(pos, cells, h_rows, bt.id)],
                "score": state["score"]
                + self._score_block(bt, cells)
                + self._type_score(bt.id, state)
                + gap_score,
                "gaps": state["gaps"] + [(gap_start, end)],
                "types_used": state["types_used"] | {bt.id, 0},
            }

        return {
            "pos": pos + cells,
            "placements": state["placements"] + [(pos, cells, h_rows, bt.id)],
            "score": state["score"]
            + self._score_block(bt, cells)
            + self._type_score(bt.id, state),
            "gaps": state["gaps"],
            "types_used": state["types_used"] | {bt.id},
        }

    def _can_fill_gap_with_mono(self, row: int, gap_start: int, gap_end: int) -> bool:
        """Check if gap at wall edge can be filled with monolith."""
        # Monolith at left wall edge: only if blocked on any row
        if gap_start == 0:
            if self.blocked[row, 0] != 1:
                return False

        if gap_end == self.num_cells:
            if self.blocked[row, gap_end - 1] != 1:
                return False

        return True

    def _compute_gap_score(
        self,
        state: Dict,
        gap_size: int,
        gap_start: int = -1,
        gap_end: int = -1,
    ) -> float:
        """Compute score penalty for monolith gap."""
        mono_penalty = gap_size * 0.5
        type_penalty = 0 if 0 in state["types_used"] else -5.0

        # Prefer gap near blocked zones (any row)
        position_bonus = 0.0
        near_blocked_right = gap_end >= 0 and self._has_blocked_near(gap_end, "right")
        near_blocked_left = gap_start >= 0 and self._has_blocked_near(gap_start, "left")

        if near_blocked_right or near_blocked_left:
            position_bonus = 5.0
        else:
            position_bonus = -10.0

        # Penalty for multiple gaps — prefer consolidating monolith
        multi_gap_penalty = -20.0 * len(state["gaps"])

        return -mono_penalty + type_penalty + position_bonus + multi_gap_penalty

    def _type_score(self, type_id: int, state: Dict) -> float:
        """Bonus for reusing types."""
        return 5.0 if type_id in state["types_used"] else -5.0

    def _try_skip_cell(self, state: Dict, row: int, pos: int) -> Optional[Dict]:
        """Try skipping one cell (will be monolith)."""
        if not self._can_place_mono_at(row, pos):
            return None

        skip_penalty = -5 if 0 in state["types_used"] else -15
        return {
            "pos": pos + 1,
            "placements": state["placements"],
            "score": state["score"] + skip_penalty,
            "gaps": state["gaps"] + [(pos, pos + 1)],
            "types_used": state["types_used"] | {0},
        }

    def _apply_solution(self, row: int, solution: Dict):
        """Apply beam search solution to grid."""
        for pos, cells, h_rows, type_id in solution["placements"]:
            self._place_block(row, pos, cells, h_rows, type_id)

        for gap_start, gap_end in solution["gaps"]:
            self._fill_monolith(row, gap_start, gap_end)

    # ============================================================
    # Scoring
    # ============================================================

    def _score_block(self, bt: BlockType, cells: int) -> float:
        """
        Score prioritizes fewer blocks (crane lifts):
        - 1 large block + monolith better than 3 small blocks
        """
        if bt.id == 0:
            return -0.5 * cells

        block_penalty = 50
        size_bonus = cells * 1.0
        height_bonus = 30 if bt.height == 600 else 0

        return size_bonus - block_penalty + height_bonus

    # ============================================================
    # Validation checks
    # ============================================================

    def _is_valid_placement(
        self, row: int, start: int, cells: int, h_rows: int, block_height: int
    ) -> bool:
        """Check if placement satisfies all constraints."""
        return (
            self._check_empty(row, start, cells, h_rows)
            and self._check_support(row, start, cells)
            and self._check_seam(row, start, cells, block_height)
        )

    def _check_empty(self, row: int, start: int, cells: int, h_rows: int) -> bool:
        return np.all(self.grid[row : row + h_rows, start : start + cells] == 0)

    def _check_support(self, row: int, start: int, cells: int) -> bool:
        if row == 0:
            return True
        below = self.grid[row - 1, start : start + cells]

        return np.any(below != 0)

    def _check_seam(self, row: int, start: int, cells: int, block_height: int) -> bool:
        """Check seam alignment constraint."""
        if row == 0:
            return True

        min_shift_mm = block_height * self.min_seam_shift_ratio
        min_shift_cells = int(min_shift_mm // self.grid_step)
        seams_below = self._find_seams(row - 1)

        left = start
        right = start + cells

        for seam in seams_below:
            if (
                abs(seam - left) < min_shift_cells
                or abs(seam - right) < min_shift_cells
            ):
                return False

        return True

    def _find_seams(self, row: int) -> List[int]:
        """Find block boundaries, ignoring monolith."""
        seams = []
        for i in range(self.num_cells - 1):
            id_left = self.grid[row, i]
            id_right = self.grid[row, i + 1]

            if id_left <= 0 or id_right <= 0 or id_left == id_right:
                continue

            type_left = self.instances.get(id_left, {}).get("type_id", -1)
            type_right = self.instances.get(id_right, {}).get("type_id", -1)

            if type_left == 0 and type_right == 0:
                continue

            seams.append(i + 1)

        return seams

    # ============================================================
    # Placement
    # ============================================================

    def _place_block(self, row: int, start: int, cells: int, h_rows: int, type_id: int):
        """Place block on grid."""
        inst_id = self.instance_counter
        self.instance_counter += 1

        actual_overhang_left = max(0, -start)
        actual_overhang_right = max(0, (start + cells) - self.num_cells)

        self.instances[inst_id] = {
            "row": row,
            "start_cell": start,
            "end_cell": start + cells,
            "h_rows": h_rows,
            "type_id": type_id,
            "overhang_left_mm": actual_overhang_left * self.grid_step,
            "overhang_right_mm": actual_overhang_right * self.grid_step,
        }

        grid_start = max(0, start)
        grid_end = min(self.num_cells, start + cells)
        if grid_start < grid_end:
            self.grid[row : row + h_rows, grid_start:grid_end] = inst_id

    # ============================================================
    # Openings
    # ============================================================

    def _apply_openings(self, openings: List[Opening]):
        for op in openings:
            x0 = max(0, (op.center_x - op.width // 2) // self.grid_step)
            x1 = min(self.num_cells, (op.center_x + op.width // 2) // self.grid_step)
            y0 = max(0, (op.center_y - op.height // 2) // self.row_height)
            y1 = min(
                self.num_rows,
                (op.center_y + op.height // 2 + self.row_height - 1) // self.row_height,
            )

            self.blocked[y0:y1, x0:x1] = 1
            self.grid[y0:y1, x0:x1] = -1

    def _get_free_segments(self, row: int) -> List[Tuple[int, int]]:
        """Get continuous free segments in row."""
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
