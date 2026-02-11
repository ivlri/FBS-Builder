from typing import List, Dict, Any
from collections import defaultdict

from src.builder.structures import WallInstance, GRID_STEP
from src.builder.fbs_builder import BLOCK_TYPES
from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner
from src.optimizer.bonding_optimizer import BondingOptimizer, OptimizationResult

BLOCK_NAMES = {bt.id: bt.name for bt in BLOCK_TYPES}


def _merge_consecutive(blocks: List[Dict]) -> List[Dict]:
    """Merge consecutive blocks of same type."""
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


def format_layers(instances: Dict[str, Dict], grid_step: int = 20) -> Dict[str, Any]:
    """Format instances by construction layers (600mm each)."""
    if not instances:
        return {"text": "", "layers": []}

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
            "start_mm": inst["start"] * grid_step,
            "h_rows": h_rows
        }

        if h_rows == 2:
            layers_data[layer]["row0"].append(block_info)
            layers_data[layer]["row1"].append(block_info)
        else:
            row_in_layer = row % 2
            layers_data[layer][f"row{row_in_layer}"].append(block_info)

    lines = []
    layers_output = []

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
            layers_output.append(row0)
        else:
            if row0:
                merged = _merge_consecutive(row0)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"  L{layer}(row0) | {text}")
            if row1:
                merged = _merge_consecutive(row1)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"  L{layer}(row1) | {text}")
            layers_output.append({"row0": row0, "row1": row1})

    return {
        "text": "\n".join(lines),
        "layers": layers_output
    }


def format_detailed(instances: Dict[str, Dict], grid_step: int = 20) -> str:
    """Format with positions for Revit transfer."""
    if not instances:
        return "  (empty)"

    lines = []
    layers_data = defaultdict(list)

    for inst_id, inst in instances.items():
        row = inst["row"]
        layer = row // 2
        h_rows = inst.get("h_rows", 1)
        length_mm = (inst["end"] - inst["start"]) * grid_step
        start_mm = inst["start"] * grid_step
        type_id = inst["type_id"]

        layers_data[layer].append({
            "type_id": type_id,
            "name": BLOCK_NAMES.get(type_id, f"Type{type_id}"),
            "start_mm": start_mm,
            "length_mm": length_mm,
            "h_rows": h_rows,
        })

    for layer in sorted(layers_data.keys()):
        blocks = sorted(layers_data[layer], key=lambda x: x["start_mm"])
        blocks = _merge_consecutive_detailed(blocks)

        block_strs = []
        for b in blocks:
            if b["type_id"] == 0:
                block_strs.append(f"M({b['length_mm']}mm)@{b['start_mm']}")
            else:
                block_strs.append(f"{b['name']}@{b['start_mm']}")

        lines.append(f"  L{layer}: {', '.join(block_strs)}")

    return "\n".join(lines)


def _merge_consecutive_detailed(blocks: List[Dict]) -> List[Dict]:
    """Merge consecutive monolith blocks."""
    if not blocks:
        return []
    merged = []
    current = dict(blocks[0])
    for b in blocks[1:]:
        # Only merge monolith (type_id=0)
        if b["type_id"] == 0 and current["type_id"] == 0:
            if b["start_mm"] == current["start_mm"] + current["length_mm"]:
                current["length_mm"] += b["length_mm"]
                continue
        merged.append(current)
        current = dict(b)
    merged.append(current)
    return merged


def print_wall_result(wall_id: int, wall: WallInstance, instances: Dict, reward: float):
    """Print single wall result."""
    print(f"\n{'-'*50}")
    print(f"Wall {wall_id}: {wall.length}mm x {wall.height}mm (weight {wall.weight}mm)")
    print(f"Reward: {reward:.2f}")
    print('-'*50)

    output = format_layers(instances, grid_step=GRID_STEP)
    print(output["text"])

    # Statistics
    stats = compute_stats(instances, wall.num_cells, GRID_STEP)
    print("\n  Stats:")
    print(f"    FBS blocks: {stats['fbs_count']} ({stats['fbs_percent']:.1f}%)")
    print(f"    Monolith: {stats['monolith_mm']}mm ({stats['monolith_percent']:.1f}%)")


def compute_stats(instances: Dict, num_cells: int, grid_step: int) -> Dict:
    """Compute layout statistics."""
    if not instances:
        return {"fbs_count": 0, "fbs_percent": 0, "monolith_mm": 0, "monolith_percent": 0}

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


def run_default(walls: List[WallInstance], runner: ModelRunner, context: ContextBuilder) -> List[Dict]:
    """Run with default bonding (no optimization)."""
    results = []

    for i, wall in enumerate(walls):
        context_data = {
            "walls": walls,
            "current_idx": i
        }

        result = runner.run(
            wall=wall,
            context_builder=context,
            context_data=context_data,
        )

        results.append({
            "wall_id": wall.id,
            "wall": wall,
            "instances": result.get("instances", {}),
            "reward": result.get("reward", 0),
            "grid": result.get("grid"),
        })

    return results


def run_optimized(walls: List[WallInstance], runner: ModelRunner, context: ContextBuilder) -> OptimizationResult:
    """Run with BondingOptimizer."""
    optimizer = BondingOptimizer(runner, context)
    return optimizer.optimize_chain(walls)


def test_3_walls_comparison():
    """Compare default vs optimized on 3 walls."""
    print("=" * 60)
    print("TEST: Compare default vs optimized")
    print("=" * 60)

    walls = [
        WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    ]

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    context = ContextBuilder(grid_step=GRID_STEP)

    # === DEFAULT ===
    print("\n" + "#" * 60)
    print("#  DEFAULT BONDING (no optimization)")
    print("#" * 60)

    default_results = run_default(walls, runner, context)
    default_total_reward = 0

    for res in default_results:
        print_wall_result(res["wall_id"], res["wall"], res["instances"], res["reward"])
        default_total_reward += res["reward"]

    print(f"\n{'-'*50}")
    print(f"TOTAL DEFAULT: reward = {default_total_reward:.2f}")

    # === OPTIMIZED ===
    print("\n" + "#" * 60)
    print("#  OPTIMIZED BONDING")
    print("#" * 60)

    opt_result = run_optimized(walls, runner, context)

    print(f"\nBonding assignments: {opt_result.bonding_assignments}")
    print(f"RL calls: {opt_result.num_rl_calls}")

    opt_total_reward = 0
    for wall in walls:
        wall_result = opt_result.wall_results.get(wall.id)
        if wall_result:
            print_wall_result(wall.id, wall, wall_result.instances, wall_result.reward)
            opt_total_reward += wall_result.reward

    print(f"\n{'-'*50}")
    print(f"TOTAL OPTIMIZED: reward = {opt_total_reward:.2f}")

    # === COMPARISON ===
    print("\n" + "#" * 60)
    print("#  COMPARISON")
    print("#" * 60)
    diff = opt_total_reward - default_total_reward
    print(f"\n  Default:   {default_total_reward:.2f}")
    print(f"  Optimized: {opt_total_reward:.2f}")
    print(f"  Diff:      {diff:+.2f} ({diff/abs(default_total_reward)*100 if default_total_reward else 0:+.1f}%)")


def test_4_walls_chain():
    """Test 4 walls chain."""
    print("\n\n" + "=" * 60)
    print("TEST: Chain of 4 walls")
    print("=" * 60)

    walls = [
        WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=2400, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=3600, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=4, length=2700, height=1800, weight=300, grid_step=GRID_STEP),
    ]

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    context = ContextBuilder(grid_step=GRID_STEP)

    opt_result = run_optimized(walls, runner, context)

    print(f"\nBonding: {opt_result.bonding_assignments}")
    print(f"Total score: {opt_result.total_score:.2f}")
    print(f"RL calls: {opt_result.num_rl_calls}")

    for wall in walls:
        wall_result = opt_result.wall_results.get(wall.id)
        if wall_result:
            print_wall_result(wall.id, wall, wall_result.instances, wall_result.reward)


if __name__ == "__main__":
    test_3_walls_comparison()
    # test_different_lengths()
    # test_4_walls_chain()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
