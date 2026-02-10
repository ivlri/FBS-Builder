from typing import List, Tuple, Dict, Any, Set, Optional
from src.builder.structures import WallInstance, GRID_STEP
import numpy as np
from collections import defaultdict

from src.builder.structures import WallInstance, Opening, GRID_STEP
from src.builder.fbs_builder import BLOCK_TYPES

from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner


def _merge_consecutive(blocks: List[Dict]) -> List[Dict]:
    """Merge consecutive blocks of same type (especially monolith)."""
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
    """
    Format instances by construction layers (600mm each).

    Returns:
        {
            "text": "L0 | 3(1200mm), 4(900mm)
                     L1 | ...",
            "layers": [[{type_id, length_mm, start, end}, ...], ...]
        }
    """
    if not instances:
        return {"text": "", "layers": []}

    # Group by layer (layer = row // 2)
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
            lines.append(f"L{layer} | {text}")
            layers_output.append(row0)
        else:
            if row0:
                merged = _merge_consecutive(row0)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"L{layer}(1-300mm) | {text}")
            if row1:
                merged = _merge_consecutive(row1)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"L{layer}(2-300mm) | {text}")
            layers_output.append({"row0": row0, "row1": row1})

    return {
        "text": "\n".join(lines),
        "layers": layers_output
    }

walls = [
    WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    WallInstance(id=2, length=3000, height=1800, weight=300, grid_step=GRID_STEP)
]

context = ContextBuilder(grid_step=GRID_STEP)
runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")


for i, wall in enumerate(walls):
    # grid = context.build_grid(walls, i, 3, 150)
    # for layer_idx, layer in enumerate(grid):
    #     print(f"L{layer_idx}| {''.join(map(str, layer))}")
    # print('----------------')
    context_data = {
        "walls": walls,
        "current_idx": i
    }

    result = runner.run(
        wall=wall,
        context_builder=context,
        context_data=context_data
    )

    output = format_layers(result.get("instances"), grid_step=GRID_STEP)
    print(output["text"])
    print("---")
    
# res = runner.run(w)
# for i in res.get('grid'):
#     print(i)

# print(res.get("instances"))
