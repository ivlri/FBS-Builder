# Changelog

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
