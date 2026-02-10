# Changelog

## [2026-02-10] - Bonding Optimization System

### Added

#### Новые модули

- **`src/optimizer/bonding_optimizer.py`** — Оптимизатор выбора перевязки
  - Класс `BondingOptimizer` с DP для деревьев и Beam Search для циклов
  - Многократный запуск RL inference для оценки вариантов
  - Метрика качества: % FBS, штраф за монолит, штраф за стыки
  - Dataclasses: `WallResult`, `OptimizationResult`

- **`src/optimizer/__init__.py`** — Экспорт модуля

- **`src/planner/wall_planner.py`** — Обёртка над planer.py
  - Класс `WallPlanner` для построения графа стен
  - Класс `WallData` для входных данных стен
  - Методы `get_adjacency()` и `get_adjacency_from_graph()`
  - Конвертация в `WallInstance` для optimizer

#### Документация

- **`docs/bonding_optimization_architecture.md`** — Архитектура системы оптимизации
  - Описание проблемы необратимых решений
  - Схема работы DP и Beam Search
  - Примеры с диаграммами
  - Результаты тестирования

#### Тесты

- **`src/tests/test_bonding_optimizer.py`** — Тесты оптимизатора
  - `test_chain_3_walls()` — базовый тест цепи
  - `test_compare_with_default()` — сравнение с default bonding
  - `test_different_wall_lengths()` — короткие стены

- **`src/tests/test_planner_optimizer_integration.py`** — Интеграционные тесты
  - `test_simple_chain()` — цепь 4 стен
  - `test_l_shape()` — L-образная конфигурация
  - `test_t_junction()` — T-образный стык
  - `test_real_data_subset()` — данные из planer.py
  - `test_compare_chain_vs_graph_adjacency()` — сравнение методов adjacency

---

### Changed

#### `src/contextbuilder/contextbuilder.py`

**До:**
```python
def _apply_end_restrictions(self, grid, completed_wall, next_wall):
    # Жёстко закодированный паттерн: block % 2 != 1 / block % 2 == 1
```

**После:**
```python
def build_grid(self, walls, current_idx, num_rows, num_cells,
               bonding_left=None, bonding_right=None, context_data=None):
    # Параметризованный выбор перевязки

def build_grid_with_bonding(self, wall, left_wall, right_wall,
                            bonding_left, bonding_right):
    # Упрощённый интерфейс для BondingOptimizer

def _apply_end_restrictions(self, grid, completed_wall, next_wall,
                            bonding_left=None, bonding_right=None):
    # bonding_left/right: 0 = block even layers, 1 = block odd layers
```

#### `src/builder/fbs_builder.py`

**Изменение в методе reset() (строка ~943):**
```python
# До:
context_mask = self.context_builder.build_grid(
    walls=self.context_data["walls"],
    current_idx=self.context_data["current_idx"],
    num_rows=self.num_rows,
    num_cells=self.num_cells,
)

# После:
context_mask = self.context_builder.build_grid(
    walls=self.context_data["walls"],
    current_idx=self.context_data["current_idx"],
    num_rows=self.num_rows,
    num_cells=self.num_cells,
    context_data=self.context_data,  # <- Добавлено
)
```

---

### Результаты тестирования

```
Test: Compare optimizer vs default (3 walls)
  Default:   142.36
  Optimized: 144.20
  Improvement: +1.3%

Test: Simple chain (4 walls)
  Total score: 235.43
  RL calls: 12

Test: T-junction (3 walls)
  Total score: 172.45
  RL calls: 14
```

---

### Архитектура

```
Planner (planer.py / wall_planner.py)
    │
    ▼
WallPlanner.process()
    │ - Строит граф
    │ - Обрабатывает T-joints
    │ - Определяет порядок обхода
    ▼
BondingOptimizer.optimize()
    │ - DP для деревьев / Beam Search для циклов
    │ - Многократный RL inference для оценки
    │ - Выбор лучшей комбинации bonding
    ▼
Результат: bonding_assignments + готовые раскладки
```

---

### Файловая структура (новое)

```
src/
├── optimizer/
│   ├── __init__.py              # NEW
│   └── bonding_optimizer.py     # NEW
├── planner/
│   ├── planer.py                # существующий
│   └── wall_planner.py          # NEW
├── contextbuilder/
│   └── contextbuilder.py        # MODIFIED
├── builder/
│   └── fbs_builder.py           # MODIFIED
└── tests/
    ├── test_bonding_optimizer.py              # NEW
    └── test_planner_optimizer_integration.py  # NEW

docs/
└── bonding_optimization_architecture.md       # NEW
```
