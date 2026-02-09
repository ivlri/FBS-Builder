from typing import List, Tuple, Dict, Any, Set, Optional
from src.builder.structures import WallInstance, GRID_STEP
import numpy as np
from collections import defaultdict

from src.builder.structures import WallInstance, Opening,  GRID_STEP

from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner
walls = [
    WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    WallInstance(id=1, length=3000, height=1800, weight=200, grid_step=GRID_STEP),
    WallInstance(id=2, length=3000, height=1800, weight=300, grid_step=GRID_STEP)
]

context = ContextBuilder(grid_step=GRID_STEP)
runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")

def compact_rows(res):
    rows = defaultdict(list)

    for seg in res.values():
        rows[seg["row"]].append(seg)

    for row in sorted(rows):
        segments = sorted(rows[row], key=lambda x: x["start"])

        parts = []
        prev_type = None

        for s in segments:
            current_type = s["type_id"]

            if current_type != prev_type:
                parts.append(str(current_type))
                prev_type = current_type

        print(f"row {row}: " + " | ".join(parts))

for i, wall in enumerate(walls):
    # grid = context.build_grid(walls, i)
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

    # compact_rows(result.get("instances"))
    
# res = runner.run(w)
# for i in res.get('grid'):
#     print(i)

# print(res.get("instances"))
