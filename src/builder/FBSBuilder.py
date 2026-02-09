# fbs_builder_env.py
# Stage 3: Single wall with openings (blocked zones)
# Grid resolution: 300mm rows (600mm blocks span 2 rows, 300mm blocks span 1 row)
import math
from typing import List, Tuple, Dict, Any, Set, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
from .structures import BlockType, WallInstance, Opening
from src.contextbuilder.contextbuilder import ContextBuilder


# === Default block types ===
BLOCK_TYPES = [

    # Monolith
    BlockType(id=0, length=20,   height=300, name="Монолит 300"),

    # FBS 600mm
    BlockType(id=2, length=2400, height=600, name="ФБС 24.6"),
    BlockType(id=3, length=1200, height=600, name="ФБС 12.6"),
    BlockType(id=4, length=900,  height=600, name="ФБС 9.6"),

    # FBS 300mm
    BlockType(id=5, length=2400, height=300, name="ФБС 24.3"),
    BlockType(id=6, length=1200, height=300, name="ФБС 12.3"),
    BlockType(id=7, length=900,  height=300, name="ФБС 9.3")

]

MAX_HALF_PROX = 60 


#============================================================
#Environment
#============================================================
class FBSBuilderEnv(gym.Env):
    """
    Gymnasium environment for FBS block placement.
    Stage 3: Domain randomization + openings (blocked zones).
    Grid uses 300mm rows. 600mm blocks occupy 2 rows, 300mm blocks occupy 1 row.
    """
    metadata = {"render_modes": ["human", "terminal"], "render_fps": 1}

    def __init__(
        self,
        wall_instance: WallInstance = None,
        context_builder: Optional[ContextBuilder] = None,
        context_data: dict = None,

        block_types: List[BlockType] = None,
        render_mode: str = None,
        max_steps: int = 1000,
        randomize: bool = False,
        min_length: int = 1200,
        max_length: int = 6000,
        min_height: int = 1200,
        max_height: int = 3000,
        grid_step: int = 20,
        openings: List[Opening] = None,
    ):
        super().__init__()
        """
        Initialize FBS Builder Environment.
        
        Args:
            wall_instance: Конкретная стена для строительства. Если None, используется 
                стена по умолчанию (3000x1800mm) или рандомизируется при randomize=True.
            
            block_types: Список доступных типов блоков. Если None, используются
                стандартные блоки из BLOCK_TYPES (Монолит 300, ФБС 24/12/9 для обеих высот).
            
            render_mode: Режим отображения:
                - None: без рендеринга
                - "terminal": отображение instance ID блоков
                - "terminal_human": отображение type ID блоков
            
            max_steps: Максимальное количество шагов в эпизоде (по умолчанию 1000).
                При превышении эпизод завершается с truncated=True.
            
            randomize: Режим domain randomization:
                - False: фиксированные размеры стены из wall_instance
                - True: случайные размеры стены в каждом эпизоде (для обобщения модели)
            
            min_length: Минимальная длина стены при randomize=True (по умолчанию 1200мм).
            
            max_length: Максимальная длина стены при randomize=True (по умолчанию 6000мм).
                Также определяет размер observation/action space (padding target).
            
            min_height: Минимальная высота стены при randomize=True (по умолчанию 1200мм).
                Должна быть кратна 600мм (размер слоя).
            
            max_height: Максимальная высота стены при randomize=True (по умолчанию 3000мм).
                Также определяет размер observation/action space (padding target).
                Должна быть кратна 600мм.
            
            grid_step: Размер ячейки сетки в миллиметрах (по умолчанию 20мм).
                Определяет разрешение горизонтальной сетки:
                - Стена 3000мм = 150 ячеек при grid_step=20
                - Блок ФБС 2400мм занимает 120 ячеек
            
            openings: Список проемов (окон/дверей) в стене. Если None и randomize=True,
                генерируются случайные проемы. Проемы блокируют зоны сетки.
        
        Attributes (инициализируемые):
            max_cells: Максимальное число ячеек по горизонтали (max_length / grid_step)
            max_rows: Максимальное число рядов по вертикали (max_height / 300)
            n_types: Количество типов блоков
            block_cells: Список размеров блоков в ячейках
            block_rows: Список высот блоков в рядах (300мм)
            max_fbs_cells: Размер самого большого ФБС блока (для нормализации награды)
            
            num_cells: Текущее число ячеек стены (переопределяется в reset при randomize)
            num_rows: Текущее число рядов стены (переопределяется в reset при randomize)
            num_layers: Число слоев по 600мм (num_rows // 2)
            
            grid: Сетка с instance ID блоков (int32, shape: max_rows x max_cells)
            grid_human: Сетка с type ID блоков для отображения (int32)
            blocked_mask: Маска заблокированных зон (проемы) (int8, 0=свободно, 1=заблокировано)
            
            current_row: Текущий ряд строительства
            step_count: Счетчик шагов в текущем эпизоде
            total_reward: Накопленная награда за эпизод
            
            instance_counter: Счетчик для уникальных ID блоков
            instances: Словарь {instance_id: metadata} размещенных блоков
            pen_bouns: Множество штрафных границ блоков
            
            n_actions: Общее число действий (n_types * max_cells + 1 для COPY_LAYER)
            COPY_ACTION_ID: ID мета-действия копирования слоя
            
            action_space: Discrete(n_actions)
            observation_space: Словарь с grid, blocked_mask, current_row, action_mask
        
        Raises:
            ValueError: Если wall_instance превышает max_cells или max_rows
        
        Notes:
            - Сетка использует разрешение 300мм по вертикали (ряды)
            - Блоки 600мм занимают 2 ряда, блоки 300мм — 1 ряд
            - Монолит (type_id=0,1) используется для заполнения малых зазоров
            - ФБС блоки (type_id=2-5) — основные строительные элементы
            - При randomize=True размеры стены генерируются случайно в каждом reset()
            - Проемы блокируют размещение блоков и влияют на допустимость 300мм блоков
        """

        self.block_types = block_types or BLOCK_TYPES
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.randomize = randomize
        self.grid_step = grid_step
        self.min_length = min_length
        self.max_length = max_length
        self.min_height = min_height
        self.max_height = max_height

        self.context_builder = context_builder
        self.context_data = context_data
        #Max dimensions define observation/action space sizes (padding target)
        self.max_cells = max_length // grid_step
        self.max_rows = max_height // 300

        self.n_types = len(self.block_types)

        #Precompute block dimensions in cells/rows
        self.block_cells = [bt.num_cells(self.grid_step) for bt in self.block_types]
        self.block_rows = [bt.num_rows() for bt in self.block_types]

        self.max_fbs_cells = max(
            bt.num_cells(self.grid_step) for bt in self.block_types if bt.id != 0
        )

        #Current wall dimensions (overwritten in reset when randomize=True)
        if wall_instance is not None and not randomize:
            self.wall_instance = wall_instance
            self.num_cells = wall_instance.num_cells
            self.num_rows = wall_instance.num_rows
        elif not randomize:
            self.wall_instance = WallInstance(id=0, length=3000, height=1800, weight=300, grid_step=grid_step)
            self.num_cells = self.wall_instance.num_cells
            self.num_rows = self.wall_instance.num_rows
        else:
            self.wall_instance = None
            self.num_cells = self.max_cells
            self.num_rows = self.max_rows

        self.num_layers = self.num_rows // 2

        #Internal state
        self.grid = None
        self.grid_human = None
        self.current_row = None
        self.step_count = 0
        self.total_reward = 0.0

        self.inst_counter = 1
        self.inst = {}
        self.pen_bounds: Set[Tuple[int, int]] = set()

        self.openings = openings
        self.blocked_mask = None

        self.n_actions = self.n_types * self.max_cells

        #COPY_LAYER meta-action: copies row-pair (layer-2) to current row-pair
        self.COPY_ACTION_ID = self.n_actions
        self.n_actions += 1  # +1 for COPY_LAYER
        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=255,
                shape=(self.max_rows, self.max_cells),
                dtype=np.int16
            ),
            "blocked_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_rows, self.max_cells),
                dtype=np.int8
            ),
            "current_row": spaces.Discrete(self.max_rows + 1),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(self.n_actions,),
                dtype=np.int8
            ),
        })

        self.reset()

    #========================================
    #Core of the learning
    #========================================

    #--------- Core block placement ---------
    def _intersects(self, row: int, start: int, end: int, h_rows: int = 1) -> bool:
        """Check if placing a block at grid[row:row+h_rows, start:end] overlaps"""
        if row + h_rows > self.num_rows:
            return True
        
        region = self.grid[row:row + h_rows, start:end]
        if np.any(region != 0):
            return True
        
        if self.blocked_mask is not None:
            if np.any(self.blocked_mask[row:row + h_rows, start:end] == 1):
                return True
            
        return False

    def _fits_bounds(self, start: int, block_cells: int, row: int = 0, h_rows: int = 1) -> bool:
        return start >= 0 and (start + block_cells) <= self.num_cells and (row + h_rows) <= self.num_rows

    def _check_bonding(self, row: int, start: int, end: int) -> bool:
        """Check support from below the bottom row of the block"""
        if row == 0:
            return True
        
        layer_below = self.grid[row - 1, start:end]
        if np.any(layer_below != 0):
            return True
        
        if self.blocked_mask is not None and np.any(self.blocked_mask[row - 1, start:end] == 1):
            return True
        
        return False
    
    def _is_on_frontier(self, row: int, start: int, end: int) -> bool:
        if start == 0:
            return True
        
        if self.grid[row, start - 1] != 0:
            return True

        if self.blocked_mask is not None and self.blocked_mask[row, start - 1] == 1:
            return True
        
        if end >= self.num_cells:
            return True
        
        if self.grid[row, end] != 0:
            return True
        
        if self.blocked_mask is not None and self.blocked_mask[row, end] == 1:
            return True
        
        return False

    def _creates_small_edge_gap(self, row: int, start: int, end: int) -> bool:
        """True if placement creates unfillable gap at wall edge or next to blocked zone"""
        min_fbs = min(self.block_cells[1:])
        r = self.grid[row]
        blocked = self.blocked_mask[row] if self.blocked_mask is not None else np.zeros(self.max_cells, dtype=np.int8)

        if end < self.num_cells:
            right_boundary = self.num_cells
            for i in range(end, self.num_cells):
                if blocked[i] == 1:
                    right_boundary = i
                    break
            if np.all(r[end:right_boundary] == 0):
                right_gap = right_boundary - end
                if 0 < right_gap < min_fbs:
                    return True

        if start > 0:
            left_boundary = 0
            for i in range(start - 1, -1, -1):
                if blocked[i] == 1:
                    left_boundary = i + 1
                    break
            if np.all(r[left_boundary:start] == 0):
                left_gap = start - left_boundary
                if 0 < left_gap < min_fbs:
                    return True

        return False

    
    def _find_block_bound(self, row: int) -> np.ndarray:
        """Find boundaries between blocks in a row (not between monolith cells)"""
        if row < 0 or row >= self.num_rows:
            return np.array([], dtype=np.int32)

        bounds = []

        for i in range(self.num_cells - 1):
            if self.blocked_mask is not None:
                if self.blocked_mask[row, i] == 1 or self.blocked_mask[row, i + 1] == 1:
                    continue

            curr = self.grid_human[row, i]
            next_val = self.grid_human[row, i + 1]

            if curr == 0 or next_val == 0:
                curr_filled = self.grid[row, i] != 0
                next_filled = self.grid[row, i + 1] != 0
                if not curr_filled or not next_filled:
                    continue

            curr_inst = self.grid[row, i]
            next_inst = self.grid[row, i + 1]

            if curr_inst != next_inst:
                curr_type = self.grid_human[row, i]
                next_type = self.grid_human[row, i + 1]
                if curr_type != 0 or next_type != 0:
                    bounds.append(i + 1)

        return np.array(bounds, dtype=np.int32)
    
    #--------- Core penalty ---------
    def _bonding_block_pen(self, row: int, start: int, end: int, block_height: int) -> float:
        """Penalty for vertical seams too close between rows"""
        if row == 0:
            return 0.0

        min_bound = 0.4 * block_height
        min_cells = int(min_bound // self.grid_step)

        seams_below = self._find_block_bound(row - 1)
        if len(seams_below) == 0:
            return 0.0

        internal_seams_below = seams_below[
            (seams_below >= min_cells) & (seams_below <= self.num_cells - min_cells)
        ]
        if len(internal_seams_below) == 0:
            return 0.0

        penalty = 0.0
        new_bounds = []
        if start >= min_cells:
            new_bounds.append(start)
        if end <= self.num_cells - min_cells:
            new_bounds.append(end)

        for bound in new_bounds:
            bound_key = (row, bound)

            if bound_key in self.pen_bounds:
                continue

            d = np.min(np.abs(internal_seams_below - bound))
            if d < min_cells:
                penalty += 2.0
                self.pen_bounds.add(bound_key)

        return penalty

    def _big_mon_penalty(self, row: int) -> float:
        """Penalty for large monolith sections in a row"""
        current_row = self.grid_human[row, :self.num_cells]
        penalty = 0.0
        gaps = []
        i = 0
        n = len(current_row)

        while i < n:
            if current_row[i] == 0:
                j = i
                while j < n and current_row[j] == 0:
                    j += 1
                gaps.append(j - i)
                i = j
            else:
                i += 1

        min_block_cells = min(self.block_cells[1:])
        for g in gaps:
            if g > min_block_cells:
                penalty += 1.0 * (g / min_block_cells)

        return penalty
    
    #--------- Core reward Calculations ---------
    def _calc_continuity_bonus(self, row: int, start: int, end: int) -> float:
        """Bonus for placing blocks adjacent to existing blocks"""
        bonus = 0.0
        r = self.grid[row]

        if start > 0 and r[start - 1] != 0:
            bonus += 2.0
        if end < self.num_cells and r[end] != 0:
            bonus += 2.0

        return bonus

    def _find_filled(self, row: np.ndarray, start: int) -> int:
        """Find nearest filled cell to the left."""
        for i in range(start - 1, -1, -1):
            if row[i] != 0:
                return i
        return -1

    def _calc_gap_pen(self, row: int, start: int, end: int) -> float:
        """Penalty for creating small unfillable gaps"""
        penalty = 0.0
        row_inst = self.grid[row]

        if start > 0:
            left_filled = self._find_filled(row_inst, start)

            if left_filled == -1:
                gap_size = start
            else:
                gap_size = start - left_filled - 1

            min_fbs_cells = min(self.block_cells[1:])
            if 0 < gap_size < min_fbs_cells:
                penalty += 1.0 * (1.0 - gap_size / min_fbs_cells)

        if end < self.num_cells:
            right_filled = self._find_filled(row_inst, end)

            if right_filled == -1:
                gap_size = self.num_cells - end
            else:
                gap_size = right_filled - end

            min_fbs_cells = min(self.block_cells[1:])
            if 0 < gap_size < min_fbs_cells:
                penalty += 1.0 * (1.0 - gap_size / min_fbs_cells)

        return penalty

    def _calc_edge_bonus(self, row: int, start: int, end: int) -> float:
        """Bonus for placing blocks at wall edges"""
        bonus = 0.0
        if start == 0:
            bonus += 1.5

        if end == self.num_cells:
            bonus += 1.5

        edge_zone = self.num_cells // 10
        if start > 0 and start <= edge_zone:
            bonus += 0.5

        if end < self.num_cells and end >= self.num_cells - edge_zone:
            bonus += 0.5

        return bonus

    def _check_seam_alignment(self, 
                              row: int, 
                              start: int, 
                              end: int, 
                              block_height: int = 600) -> bool:
        """
        Check that new block doesn't create a seam too close to seams in the row below.
        min scales with block height: 0.4 * block_height
        """
        if row == 0:
            return True

        min_bound = 0.4 * block_height
        min_cells = int(min_bound // self.grid_step)

        seams_below = self._find_block_bound(row - 1)
        if len(seams_below) == 0:
            return True

        internal_seams_below = seams_below[
            (seams_below >= min_cells) & (seams_below <= self.num_cells - min_cells)
        ]
        if len(internal_seams_below) == 0:
            return True

        new_boundaries = []
        if start >= min_cells:
            new_boundaries.append(start)
        if end <= self.num_cells - min_cells:
            new_boundaries.append(end)

        if len(new_boundaries) == 0:
            return True

        for boundary in new_boundaries:
            d = np.min(np.abs(internal_seams_below - boundary))
            if d < min_cells:
                return False

        return True
    
    #========================================
    #Openings
    #========================================
    def _apply_openings(self, openings: List[Opening]):
        """Mark opening zones as blocked in grid and blocked_mask (300mm row resolution)"""
        for op in openings:
            half_w = op.width // 2
            half_h = op.height // 2

            x_start = max(0, op.center_x - half_w)
            x_end = min(self.num_cells * self.grid_step, op.center_x + half_w)
            y_start = max(0, op.center_y - half_h)
            y_end = min(self.num_rows * 300, op.center_y + half_h)

            #Convert to grid coordinates
            col_start = x_start // self.grid_step
            col_end = x_end // self.grid_step
            row_start = y_start // 300
            row_end = math.ceil(y_end / 300)

            row_end = min(row_end, self.num_rows)
            col_end = min(col_end, self.num_cells)

            #Mark as blocked       
            for r in range(row_start, row_end):
                self.blocked_mask[r, col_start:col_end] = 1
                self.grid[r, col_start:col_end] = -1
                self.grid_human[r, col_start:col_end] = 1

    def _gen_random_openings(self) -> List[Opening]:
        """Generate 0-2 random openings for randomization"""
        if self.np_random.random() < 0.5:
            return []

        wall_w = self.num_cells * self.grid_step
        wall_h = self.num_rows * 300
        min_margin = 900

        n_openings = int(self.np_random.integers(1, 3))
        result = []

        for _ in range(n_openings):
            w = int(self.np_random.integers(200, 601))
            h = int(self.np_random.integers(200, 601))

            if wall_w < w + 2 * min_margin or wall_h < h:
                continue

            x_min = min_margin + w // 2
            x_max = wall_w - min_margin - w // 2
            if x_min > x_max:
                continue

            y_min = h // 2
            y_max = wall_h - h // 2
            if y_min > y_max:
                continue

            x = int(self.np_random.integers(x_min, x_max + 1))
            y = int(self.np_random.integers(y_min, y_max + 1))

            new_op = Opening(center_x=x, center_y=y, width=w, height=h)

            overlap = False
            for existing in result:
                ex_left = existing.center_x - existing.width // 2
                ex_right = existing.center_x + existing.width // 2
                ey_bot = existing.center_y - existing.height // 2
                ey_top = existing.center_y + existing.height // 2

                nx_left = x - w // 2
                nx_right = x + w // 2
                ny_bot = y - h // 2
                ny_top = y + h // 2

                if not (nx_right + min_margin <= ex_left or
                        nx_left >= ex_right + min_margin or
                        ny_top <= ey_bot or ny_bot >= ey_top):
                    overlap = True
                    break

            if not overlap:
                result.append(new_op)

        return result

    #--------- Opening proximity for 300mm blocks --------- 
    def _opening_proximity_mask(self, row: int) -> np.ndarray:
        """
        Returns a boolean array of shape (num_cells,).
        True = position is within MAX_HALF_PROXIMITY cells of a blocked cell in this row.
        If no blocked cells in this row, all False (300mm blocks disallowed everywhere)
        """
        blocked_row = self.blocked_mask[row, :self.num_cells]
        blocked_positions = np.where(blocked_row == 1)[0]
        if len(blocked_positions) == 0:
            return np.zeros(self.num_cells, dtype=bool)

        near_opening = np.zeros(self.num_cells, dtype=bool)
        for bp in blocked_positions:
            left = max(0, bp - MAX_HALF_PROX)
            right = min(self.num_cells, bp + MAX_HALF_PROX + 1)
            near_opening[left:right] = True

        return near_opening
    
    #========================================
    #Action mask (legal actions)
    #========================================

    def compute_action_mask(self) -> np.ndarray:
        """Boolean mask shape (n_actions,) for Discrete action space."""
        mask = np.zeros(self.n_actions, dtype=np.int8)
        row = self.current_row

        if row >= self.num_rows:
            return mask

        # Precompute proximity mask for 300mm blocks
        proximity = self._opening_proximity_mask(row)
        has_any_opening = np.any(self.blocked_mask[:self.num_rows, :self.num_cells] == 1)

        fbs_can_fit = np.zeros(self.num_cells, dtype=bool)

        edge_gap_blocked = []
        for t_idx in range(2, self.n_types):  # Skip monolith types (0, 1)
            bt = self.block_types[t_idx]
            b_cells = self.block_cells[t_idx]
            h_rows = self.block_rows[t_idx]

            # 300mm blocks only allowed near openings (if openings exist)
            is_half_height = (bt.height == 300)
            if is_half_height and has_any_opening:
                row_proximity = proximity
            elif is_half_height and not has_any_opening:
                # No openings -> no 300mm blocks
                continue
            else:
                row_proximity = None

            for s in range(0, self.num_cells - b_cells + 1):
                if not self._fits_bounds(s, b_cells, row, h_rows):
                    continue

                # Proximity check for 300mm blocks
                if row_proximity is not None:
                    if not np.any(row_proximity[s:s + b_cells]):
                        continue

                if self._intersects(row, s, s + b_cells, h_rows):
                    continue
                if not self._check_bonding(row, s, s + b_cells):
                    continue
                if not self._check_seam_alignment(row, s, s + b_cells, bt.height):
                    continue
                if not self._is_on_frontier(row, s, s + b_cells):
                    continue

                action_idx = t_idx * self.max_cells + s
                if self._creates_small_edge_gap(row, s, s + b_cells):
                    edge_gap_blocked.append((action_idx, s, s + b_cells))
                    continue

                mask[action_idx] = 1
                fbs_can_fit[s:s + b_cells] = True

        # Safety valve: if ALL FBS actions blocked by edge gap rule, restore them
        if not np.any(mask[self.max_cells:]) and edge_gap_blocked:
            for action_idx, s, e in edge_gap_blocked:
                mask[action_idx] = 1
                fbs_can_fit[s:e] = True

        # Monolith (300mm only) — only where no FBS fits, ban positions 0 and num_cells-1
        for t_idx in [0]:
            b_cells = self.block_cells[t_idx]  # = 1
            h_rows = self.block_rows[t_idx]    # 1 for 300mm
            for s in range(self.num_cells):
                if self._intersects(row, s, s + b_cells, h_rows):
                    continue
                if not self._check_bonding(row, s, s + b_cells):
                    continue
                if not self._is_on_frontier(row, s, s + b_cells):
                    continue
                # Only allow monolith where no FBS can cover this cell
                if fbs_can_fit[s]:
                    continue
                # Ban positions 0 and num_cells-1 (edges should have FBS)
                if s == 0 or s == self.num_cells - 1:
                    continue
                action_idx = t_idx * self.max_cells + s
                mask[action_idx] = 1

        # Safety valve: if mask is completely empty, re-allow monolith at edges
        if not np.any(mask):
            for t_idx in [0]:  # Try monolith
                b_cells = self.block_cells[t_idx]
                h_rows = self.block_rows[t_idx]
                for s in [0, self.num_cells - 1]:
                    if s >= self.num_cells:
                        continue
                    if self._intersects(row, s, s + b_cells, h_rows):
                        continue
                    if not self._check_bonding(row, s, s + b_cells):
                        continue
                    if not self._is_on_frontier(row, s, s + b_cells):
                        continue
                    action_idx = t_idx * self.max_cells + s
                    mask[action_idx] = 1

        # COPY_LAYER meta-action
        if self._can_copy_layer(row):
            mask[self.COPY_ACTION_ID] = 1

        return mask

    def decode_action(self, action: int) -> Tuple[int, int]:
        if action == self.COPY_ACTION_ID:
            return -1, -1
        t_idx = action // self.max_cells
        start = action % self.max_cells
        return t_idx, start
    
    #========================================
    #Copy layer action
    #========================================
    def _is_row_pair(self, row):
        for r in [row, row + 1]:
            non_blocked = self.blocked_mask[r, :self.num_cells] == 0
            if np.any(self.grid[r, :self.num_cells][non_blocked] != 0):
                return False
        
    def _type_idx_by_id(self, type_id: int) -> int:
        """Find block_types index by type id."""
        for i, bt in enumerate(self.block_types):
            if bt.id == type_id:
                return i
        return 0
    
    def _can_copy_layer(self, row: int) -> bool:
        """
        Check if current row-pair can be filled by copying from 4 rows below (layer-2).
        Copies rows [row-4, row-3] -> [row, row+1]
        """
        if row < 4 or row + 1 >= self.num_rows:
            return False

        # Current row-pair must be empty (excluding blocked zones)
        if not self._is_row_pair(row):
            return False

        # Source row-pair must be fully filled
        if not self._is_row_pair(row - 4):
            return False

        # Check that all blocks from source rows can be placed
        for src_r_offset in [0, 1]:
            src_r = row - 4 + src_r_offset
            dst_r = row + src_r_offset
            source_instances = [
                (inst_id, meta) for inst_id, meta in self.inst.items()
                if meta["row"] == src_r
            ]
            for inst_id, meta in source_instances:
                start = meta["start"]
                end = meta["end"]
                
                if not self._check_bonding(dst_r, start, end):
                    return False
                
                block_h = (
                    self.block_types[self._type_idx_by_id(meta["type_id"])].height 
                    if meta["type_id"] != 0 
                    else 300
                )

                if not self._check_seam_alignment(dst_r, start, end, block_h):
                    return False

        return True
    
    def _execute_copy_layer(self):
        """
        Execute COPY_LAYER action: copy row-pair from 4 rows below.
        """
        row = self.current_row
        step_reward = 0.0
        info = {}

        fbs_block_count = 0

        #Copy blocks from one source row to destination row
        for src_r_offset in [0, 1]:
            src_r = row - 4 + src_r_offset
            dst_r = row + src_r_offset

            source_instances = [
                (inst_id, meta) for inst_id, meta in self.inst.items()
                if meta["row"] == src_r
            ]

            # For multi-row blocks only copy from the bottom row
            for inst_id, meta in source_instances:
                start = meta["start"]
                end = meta["end"]
                block_type_id = meta["type_id"]
                h_rows = meta["h_rows"]

                # For multi-row blocks only copy from the bottom row of the source
                if src_r_offset == 1 and h_rows > 1:
                    continue 

                dst_h_end = dst_r + h_rows
                if dst_h_end > self.num_rows:
                    continue

                # Skip if target zone is blocked
                if np.any(self.blocked_mask[dst_r:dst_r + h_rows, start:end] == 1):
                    continue

                new_instance_id = self.inst_counter
                self.inst_counter += 1

                self.inst[new_instance_id] = {
                    "type_id": block_type_id,
                    "row": dst_r,
                    "start": start,
                    "end": end,
                    "h_rows": h_rows,
                }

                self.grid[dst_r:dst_r + h_rows, start:end] = new_instance_id
                self.grid_human[dst_r:dst_r + h_rows, start:end] = block_type_id

                if block_type_id != 0:
                    fbs_block_count += 1

        # COPY_LAYER reward
        step_reward += 50.0 + 12.0 * fbs_block_count

        # Row completion bonus
        step_reward += 3.0
        blocks_in_row = len(set(self.grid[row, :self.num_cells]) - {0, -1})
        min_possible_blocks = self.num_cells // self.max_fbs_cells
        efficiency = min_possible_blocks / max(blocks_in_row, 1)
        efficiency_bonus = 5.0 * min(efficiency, 1.0)
        step_reward += efficiency_bonus

        self._advance_row()

        terminated = False
        truncated = False

        if self.current_row >= self.num_rows:
            step_reward += 20.0
            terminated = True
            info["reason"] = "all_rows_completed"
        elif self.step_count >= self.max_steps:
            truncated = True
            info["reason"] = "max_steps"

        self.total_reward += step_reward
        info["total_reward"] = self.total_reward
        info["copy_layer"] = True
        info["fbs_blocks_copied"] = fbs_block_count

        # Save terminal state before potential auto-reset by VecEnv wrapper
        if terminated or truncated:
            info["terminal_grid"] = self.grid_human.copy()
            info["terminal_instances"] = dict(self.inst)

        return self._get_obs(), step_reward, terminated, truncated, info

    def _advance_row(self):
        """Advance current_row past all fully-filled rows."""
        while self.current_row < self.num_rows:
            row_filled = (self.grid[self.current_row, :self.num_cells] != 0) | \
                         (self.blocked_mask[self.current_row, :self.num_cells] == 1)
            if np.all(row_filled):
                self.current_row += 1
            else:
                break

    #========================================
    #GYM INTERFACE
    #========================================Z
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            min_cells = self.min_length // self.grid_step
            max_cells = self.max_length // self.grid_step
            self.num_cells = int(self.np_random.integers(min_cells, max_cells + 1))

            min_rows = self.min_height // 300
            max_rows = self.max_height // 300

            # Ensure even number of rows (walls are multiples of 600mm)
            min_rows = max(min_rows, 2)
            raw_rows = int(self.np_random.integers(min_rows // 2, max_rows // 2 + 1))
            self.num_rows = raw_rows * 2

        self.num_layers = self.num_rows // 2

        # Base matrix is zeros
        self.grid = np.zeros((self.max_rows, self.max_cells), dtype=np.int32)
        self.grid_human = np.zeros((self.max_rows, self.max_cells), dtype=np.int32)
        self.blocked_mask = np.zeros((self.max_rows, self.max_cells), dtype=np.int8)

        # --- ContextBuilder override ---
        if self.context_builder is not None and self.context_data is not None:
            context_mask = self.context_builder.build_grid(**self.context_data)

            # blocked_mask - only within the limits of num_rows / num_cells
            self.blocked_mask[:self.num_rows, :self.num_cells] = context_mask

            # we mark tke blocked zones as -1
            self.grid[:self.num_rows, :self.num_cells][context_mask == 1] = -1
            self.grid_human[:self.num_rows, :self.num_cells][context_mask == 1] = 1

        self.inst_counter = 1
        self.inst = {}
        self.pen_bounds = set()

        self.current_row = 0
        self.step_count = 0
        self.total_reward = 0.0

        # Apply openings/randomize/instance restrictions
        if self.openings is not None:
            self._apply_openings(self.openings)

        elif self.randomize:
            rand_ops = self._gen_random_openings()

            if rand_ops:
                self._apply_openings(rand_ops)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        info = {}
        terminated = False
        truncated = False
        step_reward = 0.0

        step_reward -= 0.02

        action = int(action)

        #COPY_LAYER action
        if action == self.COPY_ACTION_ID:
            if self._can_copy_layer(self.current_row):
                return self._execute_copy_layer()
            else:
                step_reward = -1.0
                self.total_reward += step_reward
                return self._get_obs(), step_reward, False, False, {"invalid": "copy_not_allowed"}

        t_idx, start = self.decode_action(action)

        if not (0 <= t_idx < self.n_types):
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, False, False, {"invalid": "type_idx"}

        if not (0 <= start < self.max_cells):
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, False, False, {"invalid": "start"}

        bt = self.block_types[t_idx]
        b_cells = self.block_cells[t_idx]
        h_rows = self.block_rows[t_idx]
        end = start + b_cells
        row = self.current_row

        if row + h_rows > self.num_rows:
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, True, False, {"reason": "row_overflow"}

        # Placement checks
        if not self._fits_bounds(start, b_cells, row, h_rows):
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, False, False, {"invalid": "bounds"}

        if self._intersects(row, start, end, h_rows):
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, False, False, {"invalid": "intersects"}

        if not self._check_bonding(row, start, end):
            step_reward = -1.0
            self.total_reward += step_reward
            return self._get_obs(), step_reward, False, False, {"invalid": "bonding"}

        # Place block
        instance_id = self.inst_counter
        self.inst_counter += 1

        block_type_id = bt.id

        self.inst[instance_id] = {
            "type_id": block_type_id,
            "row": row,
            "start": start,
            "end": end,
            "h_rows": h_rows,
        }

        self.grid[row:row + h_rows, start:end] = instance_id
        self.grid_human[row:row + h_rows, start:end] = block_type_id

        #--------- REWARD SHAPING ---------
        # 1. Block size reward
        if block_type_id != 0:
            size_ratio = b_cells / self.max_fbs_cells
            block_reward = 5.0 + 15.0 * size_ratio
            step_reward += block_reward

        # 2. Continuity bonus
        continuity_bonus = self._calc_continuity_bonus(row, start, end)
        step_reward += continuity_bonus

        # 3. Gap penalty
        gap_penalty = self._calc_gap_pen(row, start, end)
        step_reward -= gap_penalty

        # 4. Edge bonus
        edge_bonus = self._calc_edge_bonus(row, start, end)
        step_reward += edge_bonus

        # 5. Bonding penalty
        bonding_penalty = self._bonding_block_pen(
            row=row, start=start, end=end, block_height=bt.height
        )
        step_reward -= bonding_penalty

        # 6. Monolith near edge penalty (gradient)
        if block_type_id == 0:
            min_fbs = min(self.block_cells[1:])
            if start < min_fbs:
                edge_ratio = 1.0 - (start / min_fbs)
                step_reward -= 10.0 * edge_ratio
            if end > self.num_cells - min_fbs:
                edge_ratio = 1.0 - ((self.num_cells - end) / min_fbs)
                step_reward -= 10.0 * edge_ratio

        # 7. Monolith-on-monolith penalty (only across layer boundaries not within same 600mm layer)
        if block_type_id == 0 and row > 0 and (row // 2) != ((row - 1) // 2):
            below_filled = self.grid[row - 1, start] != 0
            below_is_monolith = self.grid_human[row - 1, start] == 0

            if below_filled and below_is_monolith:
                step_reward -= 8.0

        #--------- Row completion ---------
        row_filled = (self.grid[row, :self.num_cells] != 0) | (self.blocked_mask[row, :self.num_cells] == 1)
        if np.all(row_filled):
            step_reward += 3.0

            blocks_in_row = len(set(self.grid[row, :self.num_cells]) - {0, -1})
            min_possible_blocks = self.num_cells // self.max_fbs_cells

            efficiency = min_possible_blocks / max(blocks_in_row, 1)
            efficiency_bonus = 5.0 * min(efficiency, 1.0)

            step_reward += efficiency_bonus

            mon_penalty = self._big_mon_penalty(row) * 0.5
            step_reward -= mon_penalty

            self._advance_row()

        #--------- Termination ---------
        action_mask = self.compute_action_mask()
        if not np.any(action_mask) and self.current_row < self.num_rows:
            step_reward -= 10.0
            terminated = True
            info["reason"] = "no_legal_moves"
        elif self.current_row >= self.num_rows:
            step_reward += 20.0
            terminated = True
            info["reason"] = "all_rows_completed"
        elif self.step_count >= self.max_steps:
            truncated = True
            info["reason"] = "max_steps"

        self.total_reward += step_reward
        info["total_reward"] = self.total_reward

        # Save terminal state before potential auto-reset by VecEnv wrapper
        if terminated or truncated:
            info["terminal_grid"] = self.grid_human.copy()
            info["terminal_instances"] = dict(self.inst)

        return self._get_obs(), step_reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        mask = self.compute_action_mask()
        type_grid = np.zeros((self.max_rows, self.max_cells), dtype=np.int16)

        for inst_id, meta in self.inst.items():
            r = meta["row"]
            start = meta["start"]
            end = meta["end"]
            h = meta["h_rows"]
            type_grid[r:r + h, start:end] = meta["type_id"]

        obs = {
            "grid": type_grid,
            "blocked_mask": self.blocked_mask.copy(),
            "current_row": np.int64(self.current_row),
            "action_mask": mask,
        }
        return obs

    def get_action_mask(self):
        return self.compute_action_mask()

    def render(self):
        if self.render_mode and "terminal" in self.render_mode:
            if "human" in self.render_mode:
                grid = self.grid_human
            else:
                grid = self.grid

            wall = self.num_cells * self.grid_step
            height = self.num_rows * 300

            print(f"\n=== Wall {wall}mm x {height}mm ({self.num_cells}c x {self.num_rows}R / {self.num_layers}L) ===")
            print(f"Row: {self.current_row}, Step: {self.step_count}, Total reward: {self.total_reward:.1f}")
            
            for r in range(self.num_rows - 1, -1, -1):
                row_data = grid[r, :self.num_cells]
                blocked = self.blocked_mask[r, :self.num_cells]
                chars = []

                for i in range(self.num_cells):
                    if blocked[i] == 1:
                        chars.append("X")
                    else:
                        chars.append(f"{int(row_data[i])}")

                layer_label = r // 2
                half = "b" if r % 2 == 0 else "t"

                print(f"R{r:2d}(L{layer_label}{half}) | " + "".join(chars))

    def close(self):
        pass


#--------- Callbacks---------
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}: reward={info['episode']['r']:.1f}, length={info['episode']['l']}")
        return True

