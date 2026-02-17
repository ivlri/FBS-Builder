# Changelog

## 2026-02-17: Циклические цепочки стен + оптимизация монолита

### 1. Поддержка циклов в SolverPipeline

**Проблема:** В L-shape и замкнутых контурах стены образуют цикл. Pipeline обрабатывал их линейно — первая и последняя стены не получали chess-pattern bonding друг с другом.

**Решение:** Параметр `is_cycle: bool` в `solve_chain()` и `_process_single_wall()`.

**Файлы:** `src/solver/solver_pipeline.py`, `src/tests/overhang_test.py`

- При `is_cycle=True`: `wall[0]` видит `wall[-1]` как left_wall, `wall[-1]` видит `wall[0]` как right_wall
- Авто-определение цикла в тестах через `planner.get_stats()["has_cycles"]`

### 2. Консолидация монолита

**Проблема:** Монолит размазывался по двум сторонам стены (gap слева + gap справа) вместо одной зоны.

**Решение:** Передача `has_mono_left`/`has_mono_right` из `_solve_segment_mixed` в beam search.

**Файл:** `src/solver/solver.py`

- `_solve_segment_mixed` определяет forced mono зоны (где FBS не помещается из-за seam constraints)
- `_solve_segment_beam` и `_build_initial_states` принимают `has_mono_right`
- Если `has_mono_right=True`, beam search не генерирует left-gap варианты — монолит консолидируется в одну зону

### 3. Запрет монолита на свободном торце

**Проблема:** Монолит разрешался на краю стены если blocked был на любом ряду, а не на текущем.

**Решение:** `_can_place_mono_at` и `_can_fill_gap_with_mono` проверяют `blocked[row, pos]` только на текущем ряду.

### 4. Исправление подсчёта монолита в summary

**Проблема:** Монолит (h_rows=1) создавал отдельные instances на row 0 и row 1 — оба попадали в summary одного слоя, удваивая длину.

**Решение:** `_format_layer_summary` пропускает h_rows=1 instances на нечётных рядах.

### 5. Scoring: предпочтение монолита рядом с blocked

**Файл:** `src/solver/solver.py`

- `_has_blocked_near(pos, side)` — проверка blocked рядом с позицией
- `_compute_gap_score` — position_bonus +5 рядом с blocked, -10 вдали; multi_gap_penalty -20 за каждый существующий gap

## 2026-02-16: Ограничение блоков 300мм в solver

### Проблема
Solver использовал блоки 300мм для обхода seam constraints — экономически невыгодно.

### Решение
Блоки 300мм разрешены ТОЛЬКО в рядах, граничащих с проёмами (±1 ряд от opening).

### Изменения в `src/solver/solver.py`

1. **Разделение блоков по высоте** (`__init__`):
```python
all_fbs = [bt for bt in block_types if bt.id != 0]
self.fbs_600 = sorted([bt for bt in all_fbs if bt.height == 600], ...)
self.fbs_300 = sorted([bt for bt in all_fbs if bt.height == 300], ...)
self.fbs_blocks = self.fbs_600  # Default: only 600mm
```

2. **Сохранение openings** для проверки proximity:
```python
self.openings = openings or []
```

3. **Новый метод `_get_allowed_blocks(row)`**:
```python
def _get_allowed_blocks(self, row: int) -> List[BlockType]:
    if not self.openings:
        return self.fbs_600
    for op in self.openings:
        y0 = (op.center_y - op.height // 2) // self.row_height
        y1 = (op.center_y + op.height // 2 + self.row_height - 1) // self.row_height
        if y0 - 1 <= row <= y1:
            return self.fbs_600 + self.fbs_300
    return self.fbs_600
```

4. **Использование в beam search**: заменено `self.fbs_blocks` на `self._get_allowed_blocks(row)` в:
   - `_can_fill_without_mono()`
   - `_compute_fbs_coverage()`
   - `_solve_segment_beam()`

### Верификация
```bash
python -m src.tests.overhang_test lshape   # Без проёмов -> нет 300мм
python -m src.tests.overhang_test opening  # С проёмом -> 300мм только у проёма
```

### Результат
- **Без проёмов:** только ФБС 600мм + монолит
- **С проёмом:** ФБС 300мм только в рядах ±1 от границ проёма
