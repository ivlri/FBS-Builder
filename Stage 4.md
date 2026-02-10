# RL FBS Builder — Архитектура системы
## Roadmap

```
Этап 1 (MVP):     Одна стена без проёмов
Этап 2:           Domain randomization (разные длины стен)
Этап 3:           Одна стена с заблокированными зонами      <- ЗАВЕРШЕН
Этап 4:           Context Builder + несколько связанных стен <- ТЕКУЩИЙ
Этап 5:           Planner с эвристикой порядка обхода
Этап 6:           Экономическая оптимизация (цены блоков)
Этап 7:           Production с confidence scoring
```

# Этап 4

## Цель

Реализовать Context Builder для перевязки смежных стен (chess-pattern bonding).

## Изменения в архитектуре

**Было:** `FBSBuilder.reset()` создаёт пустую сетку `np.zeros()`

**Стало:** `ContextBuilder.build_grid()` создаёт сетку с предзаполненными ограничениями → передаёт в `FBSBuilder`

## Правила перевязки торцов

Входные данные: `walls[]`, `current_idx`
- `completed_wall` = walls[current_idx - 1] (если есть)
- `next_wall` = walls[current_idx + 1] (если есть)

### Паттерн ограничений

Ограничения чередуются по 600мм-слоям (layer = row // 2):

| Layer | Left (from completed) | Right (towards next) |
|-------|----------------------|---------------------|
| 0     | BLOCKED              | free                |
| 1     | free                 | BLOCKED             |
| 2     | BLOCKED              | free                |
| 3     | free                 | BLOCKED             |

**Логика в коде** (`contextbuilder.py:44,54`):
```python
# Left end: block when layer % 2 == 0
if block % 2 != 1:  # layers 0, 2, 4...
    grid[layer, :cells] = 1

# Right end: block when layer % 2 == 1
if block % 2 == 1:  # layers 1, 3, 5...
    grid[layer, -cells:] = 1
```

### Пример: 3 стены подряд

```
Wall 0 (первая):    [.........|FREE]  — нет completed, нечего блокировать слева
Wall 1 (средняя):   [BLOCKED..|..BLOCKED]  — ограничения с обеих сторон (чередуются)
Wall 2 (последняя): [BLOCKED..|FREE]  — нет next, нечего блокировать справа
```

### Ширина ограничения

Ширина блокируемой зоны = `wall.weight` (толщина примыкающей стены):
- 300mm wall → 15 cells @ 20mm grid
- 200mm wall → 10 cells @ 20mm grid

## Статус реализации

- [x] `ContextBuilder.build_grid()` — создание сетки с ограничениями
- [x] `ContextBuilder._apply_end_restrictions()` — логика перевязки
- [x] `ModelRunner.run()` — интеграция с context_builder
- [x] `context_test.py` — тесты на 3 стенах


