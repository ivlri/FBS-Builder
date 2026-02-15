from typing import List, Dict
from collections import defaultdict
import numpy as np

from src.solver.solver import FBSSolver
from src.solver.solver_pipeline import SolverPipeline, normalize_instances, visualize_pipeline
from src.builder.structures import WallInstance, GRID_STEP, BLOCK_TYPES


BLOCK_NAMES = {bt.id: bt.name for bt in BLOCK_TYPES}


# ============================================================
# NORMALIZATION (solver -> RL-like format)
# ============================================================

def normalize_instances(instances: Dict) -> Dict:
    normalized = {}

    for inst_id, inst in instances.items():
        normalized[inst_id] = {
            "row": inst["row"],
            "start": inst["start_cell"],
            "end": inst["end_cell"],
            "h_rows": inst.get("h_rows", 1),
            "type_id": inst["type_id"],
        }

    return normalized


# ============================================================
# LAYER FORMAT (same as RL)
# ============================================================

def _merge_consecutive(blocks: List[Dict]) -> List[Dict]:
    if not blocks:
        return []

    merged = []
    current = dict(blocks[0])

    for b in blocks[1:]:
        if b["type_id"] == current["type_id"] and b["start"] == current["end"]:
            current["end"] = b["end"]
            current["length_mm"] += b["length_mm"]
        else:
            merged.append(current)
            current = dict(b)

    merged.append(current)
    return merged


def format_layers(instances: Dict, grid_step: int = 20) -> str:
    if not instances:
        return ""

    layers_data = defaultdict(lambda: {"row0": [], "row1": []})

    for inst in instances.values():
        row = inst["row"]
        layer = row // 2
        h_rows = inst.get("h_rows", 1)
        length_mm = (inst["end"] - inst["start"]) * grid_step

        block_info = {
            "type_id": inst["type_id"],
            "length_mm": length_mm,
            "start": inst["start"],
            "end": inst["end"],
        }

        if h_rows == 2:
            layers_data[layer]["row0"].append(block_info)
            layers_data[layer]["row1"].append(block_info)
        else:
            row_in_layer = row % 2
            layers_data[layer][f"row{row_in_layer}"].append(block_info)

    lines = []

    for layer in sorted(layers_data.keys()):
        data = layers_data[layer]

        row0 = sorted(data["row0"], key=lambda x: x["start"])
        row1 = sorted(data["row1"], key=lambda x: x["start"])

        row0_sig = [(b["type_id"], b["start"], b["end"]) for b in row0]
        row1_sig = [(b["type_id"], b["start"], b["end"]) for b in row1]

        if row0_sig == row1_sig:
            merged = _merge_consecutive(row0)
            text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
            lines.append(f"  L{layer} | {text}")
        else:
            if row0:
                merged = _merge_consecutive(row0)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"  L{layer}(row0) | {text}")

            if row1:
                merged = _merge_consecutive(row1)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"  L{layer}(row1) | {text}")

    return "\n".join(lines)


# ============================================================
# STATS + REWARD
# ============================================================

def compute_stats(instances: Dict, grid_step: int) -> Dict:
    if not instances:
        return {
            "fbs_count": 0,
            "fbs_percent": 0,
            "monolith_mm": 0,
            "monolith_percent": 0,
        }

    fbs_count = 0
    monolith_cells = 0
    fbs_cells = 0

    for inst in instances.values():
        type_id = inst["type_id"]
        length = inst["end"] - inst["start"]

        if type_id == 0:
            monolith_cells += length
        else:
            fbs_cells += length
            fbs_count += 1

    total = monolith_cells + fbs_cells

    return {
        "fbs_count": fbs_count,
        "fbs_percent": (fbs_cells / total * 100) if total > 0 else 0,
        "monolith_mm": monolith_cells * grid_step,
        "monolith_percent": (monolith_cells / total * 100) if total > 0 else 0,
    }


def compute_reward(instances: Dict) -> float:
    """
    Простейшая reward-метрика для честного сравнения.
    Поощряем FBS, штрафуем монолит.
    """

    fbs = 0
    mono = 0

    for inst in instances.values():
        length = inst["end"] - inst["start"]
        if inst["type_id"] == 0:
            mono += length
        else:
            fbs += length

    return fbs * 1.0 - mono * 0.5


# ============================================================
# PRINT (RL-like format)
# ============================================================

def print_wall_result(wall_id: int, wall: WallInstance, instances: Dict):
    reward = compute_reward(instances)

    print(f"\n{'-'*50}")
    print(f"Wall {wall_id}: {wall.length}mm x {wall.height}mm (weight {wall.weight}mm)")
    print(f"Reward: {reward:.2f}")
    print('-'*50)

    layers_text = format_layers(instances, GRID_STEP)
    print(layers_text)

    stats = compute_stats(instances, GRID_STEP)

    print("\n  Stats:")
    print(f"    FBS blocks: {stats['fbs_count']} ({stats['fbs_percent']:.1f}%)")
    print(f"    Monolith: {stats['monolith_mm']}mm ({stats['monolith_percent']:.1f}%)")


# ============================================================
# SOLVER RUN
# ============================================================

def run_solver(walls: List[WallInstance]) -> List[Dict]:

    results = []

    for wall in walls:

        solver = FBSSolver(
            wall=wall,
            block_types=BLOCK_TYPES,
            openings=getattr(wall, "openings", None),
            grid_step=GRID_STEP
        )

        grid, instances = solver.solve_wall()
        instances = normalize_instances(instances)

        results.append({
            "wall_id": wall.id,
            "wall": wall,
            "instances": instances,
            "grid": grid
        })

    return results


# ============================================================
# TEST
# ============================================================

def test_solver(walls):
    print("=" * 60)
    print("TEST: SOLVER on 3 walls")
    print("=" * 60)

    results = run_solver(walls)

    total_reward = 0

    for res in results:
        print_wall_result(res["wall_id"], res["wall"], res["instances"])
        total_reward += compute_reward(res["instances"])

    print(f"\n{'-'*50}")
    print(f"TOTAL SOLVER: reward = {total_reward:.2f}")

    print("\n" + "=" * 60)
    print("Solver test completed!")
    print("=" * 60)


def test_vsolver(walls):
    """Test SolverPipeline with connected walls."""
    print("\n" + "=" * 60)
    print("TEST: SOLVER PIPELINE (multi-wall with constraints)")
    print("=" * 60)

    pipeline = SolverPipeline(grid_step=GRID_STEP)
    result = pipeline.solve_chain(walls)

    total_reward = 0

    for wall in walls:
        wall_result = result.wall_results[wall.id]
        instances = normalize_instances(wall_result.instances)
        reward = compute_reward(instances)
        total_reward += reward

        print_wall_result(wall.id, wall, instances)

    print(f"\n{'-'*50}")
    print(f"TOTAL PIPELINE: reward = {total_reward:.2f}")
    print(f"Stats: {result.total_stats}")

    # Check constraints propagation
    print("\n" + "-" * 50)
    print("Checking constraint propagation:")

    for i in range(len(walls) - 1):
        w1 = result.wall_results[walls[i].id]
        w2 = result.wall_results[walls[i + 1].id]

        # Check if wall 2 left edge respects wall 1 right edge
        width = walls[i + 1].weight // GRID_STEP
        w1_right = w1.grid[:, -width:]
        w2_left = w2.grid[:, :width]

        # Where w1 is occupied, w2 should be empty (or blocked=-1)
        conflict = False
        for r in range(min(w1_right.shape[0], w2_left.shape[0])):
            for c in range(min(w1_right.shape[1], w2_left.shape[1])):
                if w1_right[r, c] > 0 and w2_left[r, c] > 0:
                    conflict = True
                    print(f"  CONFLICT at wall {walls[i].id}->{walls[i+1].id}, row {r}, col {c}")

        if not conflict:
            print(f"  Wall {walls[i].id} -> Wall {walls[i+1].id}: OK (no overlap)")

    # Visualization
    print("\n")
    print(visualize_pipeline(result, walls, GRID_STEP))


if __name__ == "__main__":
    walls = [
        WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=1000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    ]
    # test_solver(walls)
    test_vsolver(walls)
