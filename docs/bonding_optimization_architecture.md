# Архитектура оптимизации перевязки блоков

## Контекст проблемы

### Исходная ситуация
- RL агент обучен раскладывать блоки на ОДНОЙ стене с заданными ограничениями
- ContextBuilder формирует ограничения (шахматная перевязка на торцах)
- Planner определяет порядок обхода стен

### Проблема
Детерминированный выбор типа перевязки (0 или 1) на каждом стыке может быть **неоптимальным**:
- Неудачный выбор в начале → плохая раскладка по всей цепи
- RL агент вынужден ставить маленькие блоки / много монолита
- Нет механизма "попробовать оба варианта и выбрать лучший"

### Что такое перевязка
```
Стена Wi и Wi+1 соединяются. Два варианта:

Вариант 0: Wi блокирует слои 0,2,4... | Wi+1 блокирует слои 1,3,5...
Вариант 1: Wi блокирует слои 1,3,5... | Wi+1 блокирует слои 0,2,4...

Это влияет на то какие блоки можно положить → влияет на качество раскладки.
```

---

## Предлагаемая архитектура

```
┌─────────────────────────────────────────────────────────────┐
│  PLANNER (существующий planer.py)                           │
│  - Строит граф из координат стен                            │
│  - Обрабатывает T-joints, пересечения                       │
│  - Определяет порядок обхода стен                           │
│  - Выход: граф + порядок стен                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  BONDING OPTIMIZER (новый компонент)                        │
│  - Вход: граф + порядок стен + RL агент                     │
│  - Перебирает варианты перевязки {0, 1} на каждом стыке     │
│  - МНОГОКРАТНО запускает RL inference для оценки вариантов  │
│  - Использует DP (для деревьев) или Beam Search (для циклов)│
│  - Выход: Dict[joint_id, bonding_type] + лучшая раскладка   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT BUILDER (модифицированный)                         │
│  - Вход: стена + bonding_types для её стыков                │
│  - Формирует context_grid с ВЫБРАННОЙ шахматкой             │
│  - Параметризован: bonding_left, bonding_right ∈ {0, 1}     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RL AGENT (существующий, обученный)                         │
│  - Вход: context_grid                                       │
│  - Раскладывает блоки                                       │
│  - Выход: grid + instances + quality_score                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Ключевой момент: как работает оптимизация

> **ВАЖНО:** Оптимизация работает через МНОГОКРАТНЫЙ запуск RL inference.
> Это НЕ переобучение — это использование обученной модели для оценки разных вариантов.
> ContextBuilder сам по себе НЕ даёт оптимум — он лишь формирует constraints.
> Оптимум находится путём перебора вариантов constraints и сравнения результатов RL.

### RL агент запускается МНОГОКРАТНО

```
Для N стен с ~N стыками:
- Полный перебор: 2^N вариантов × N запусков RL = O(N × 2^N)
- DP на дереве: O(N × 4) запусков RL (для каждой стены пробуем 2×2 комбинации)
- Beam Search (k=5): O(N × k × 2) запусков RL
```

### Пример для 3 стен (цепь)

```
W0 ── W1 ── W2
   s0    s1        (s0, s1 — стыки)

Варианты: s0 ∈ {0,1}, s1 ∈ {0,1} → 4 комбинации

Оценка каждой комбинации:
┌──────────────────────────────────────────────────────────────┐
│ (s0=0, s1=0):                                                │
│   context_W0 = build(left=None, right=0)  → RL → score_W0    │
│   context_W1 = build(left=0, right=0)     → RL → score_W1    │
│   context_W2 = build(left=0, right=None)  → RL → score_W2    │
│   total_score = score_W0 + score_W1 + score_W2 = 285         │
├──────────────────────────────────────────────────────────────┤
│ (s0=0, s1=1):                                                │
│   context_W0 = build(left=None, right=0)  → RL → score_W0    │
│   context_W1 = build(left=0, right=1)     → RL → score_W1    │
│   context_W2 = build(left=1, right=None)  → RL → score_W2    │
│   total_score = 312                                          │
├──────────────────────────────────────────────────────────────┤
│ (s0=1, s1=0):                                                │
│   total_score = 298                                          │
├──────────────────────────────────────────────────────────────┤
│ (s0=1, s1=1):                                                │
│   total_score = 275                                          │
└──────────────────────────────────────────────────────────────┘

Лучший: (s0=0, s1=1) со score=312
Результат: раскладки W0, W1, W2 уже готовы от этого запуска
```

### Итого запусков RL: 3 стены × 4 комбинации = 12 запусков

---

## DP на дереве: как работает

### Идея
Вместо перебора всех 2^N комбинаций, используем структуру дерева:
- Решаем задачу рекурсивно от листьев к корню
- Для каждого узла храним лучший результат для каждого входящего типа перевязки

### Псевдокод

```python
def tree_dp(node, parent, incoming_bonding) -> (score, assignments, layouts):
    """
    Возвращает:
    - score: лучший суммарный score для поддерева
    - assignments: выбранные типы перевязки
    - layouts: готовые раскладки от RL
    """

    children = neighbors(node) - {parent}

    if not children:  # Лист
        wall = get_wall(parent, node)
        context = build_context(wall, left=incoming_bonding, right=None)
        result = RL_INFERENCE(context)  # ← Запуск RL
        return result.score, {}, {node: result.layout}

    best_score = -inf
    best_assignments = {}
    best_layouts = {}

    # Перебираем варианты перевязки для детей
    for outgoing_bondings in product([0, 1], repeat=len(children)):

        # Оцениваем текущую стену
        wall = get_wall(parent, node)
        context = build_context(wall, incoming_bonding, outgoing_bondings)
        result = RL_INFERENCE(context)  # ← Запуск RL
        wall_score = result.score

        # Рекурсивно решаем для детей
        children_score = 0
        children_assignments = {}
        children_layouts = {node: result.layout}

        for child, out_bond in zip(children, outgoing_bondings):
            c_score, c_assign, c_layouts = tree_dp(child, node, out_bond)
            children_score += c_score
            children_assignments.update(c_assign)
            children_assignments[(node, child)] = out_bond
            children_layouts.update(c_layouts)

        total = wall_score + children_score

        if total > best_score:
            best_score = total
            best_assignments = children_assignments
            best_layouts = children_layouts

    return best_score, best_assignments, best_layouts
```

### Сложность
- Для узла степени d: 2^d вариантов
- Типично d ≤ 3 (T-стык) → 8 вариантов на узел
- Итого: O(N × 2^d) запусков RL ≈ O(8N) для типичного случая

---

## Beam Search: для графов с циклами

### Когда нужен
- Граф имеет циклы (замкнутый контур стен)
- DP не применим напрямую

### Идея
- Идём по стенам в порядке обхода
- Держим top-k лучших частичных решений (beam)
- Для каждого решения пробуем оба варианта перевязки
- Отсекаем худшие, оставляем top-k

```python
def beam_search(walls, graph, beam_width=5):
    order = traverse_walls(graph)

    # beam: List[(assignments, cumulative_score, layouts)]
    beam = [({}, 0.0, {})]

    for wall_idx, wall in enumerate(order):
        candidates = []
        joints = get_joints(wall)

        for assignments, score, layouts in beam:
            # Какие стыки ещё не выбраны?
            undecided = [j for j in joints if j not in assignments]

            for bondings in product([0, 1], repeat=len(undecided)):
                new_assign = {**assignments}
                for j, b in zip(undecided, bondings):
                    new_assign[j] = b

                # Запускаем RL
                context = build_context(wall, new_assign)
                result = RL_INFERENCE(context)  # ← Запуск RL

                new_layouts = {**layouts, wall_idx: result.layout}
                candidates.append((new_assign, score + result.score, new_layouts))

        # Оставляем top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

    return beam[0]  # Лучшее решение с готовыми раскладками
```

---

## Итоговый процесс

```
1. Planner: построить граф, определить порядок
                    ↓
2. BondingOptimizer:
   - Если дерево → DP
   - Если циклы → Beam Search
   - МНОГОКРАТНЫЕ запуски RL для оценки вариантов
   - Результат: лучшие assignments + готовые layouts
                    ↓
3. Вывод: раскладки уже готовы (от лучшего варианта)
   - НЕ нужен "финальный запуск" — он уже был в процессе оптимизации
```

---

## Метрика качества (score)

```python
def compute_quality(result) -> float:
    """
    Метрика для сравнения вариантов раскладки.
    """
    grid = result['grid']
    instances = result['instances']

    # Считаем статистику
    fbs_cells = count_fbs_cells(grid)
    monolith_cells = count_monolith_cells(grid)
    total_cells = grid.size
    seam_count = count_seams(instances)

    # Взвешенная сумма
    score = (
        10 * (fbs_cells / total_cells)      # % FBS блоков
        - 5 * (monolith_cells / total_cells) # Штраф за монолит
        - 0.1 * seam_count                   # Штраф за стыки
    )

    return score
```

---

## Вычислительная сложность

| Метод | Запусков RL | Когда использовать |
|-------|-------------|-------------------|
| Полный перебор | N × 2^N | N < 10 |
| DP на дереве | ~8N | Граф — дерево (основной случай) |
| Beam Search (k=5) | ~10N | Граф с циклами |

Для 20 стен:
- Перебор: 20 × 2^20 ≈ 20M запусков — неприемлемо
- DP: ~160 запусков — OK
- Beam: ~200 запусков — OK

---

## Результаты тестирования

```
============================================================
Test: Compare optimizer vs default (3 walls chain)
============================================================

Default total score:   142.36
Optimized total score: 144.20
Improvement: +1.3%
RL calls: 8

Per-wall comparison:
  Wall 1: Default=51.07 → Optimized=60.12 (+17.7%)
  Wall 2: Default=61.64 → Optimized=54.44 (-11.7%)
  Wall 3: Default=29.65 → Optimized=29.65 (same)
```

**Вывод:** Оптимизатор находит лучшие комбинации bonding, перераспределяя качество между стенами для максимизации общего score.

---

## Реализованные компоненты

1. [x] `ContextBuilder` — добавлены параметры `bonding_left`, `bonding_right`
2. [x] `BondingOptimizer` — DP для деревьев, Beam Search для циклов
3. [x] Интеграция с `FBSBuilderEnv` — передача `context_data`
4. [x] Тесты — `src/tests/test_bonding_optimizer.py`

## Следующие шаги

1. [ ] Интеграция с `planer.py` — использовать реальный граф стен
2. [ ] Улучшить метрику качества — добавить стоимость блоков
3. [ ] Тестирование на реальных конфигурациях (10+ стен)
4. [ ] Production pipeline с полным циклом Planner → Optimizer → RL
