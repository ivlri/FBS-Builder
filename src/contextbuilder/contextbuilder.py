from typing import List, Tuple, Dict, Any, Set
from src.builder.structures import BlockType, WallInstance, Opening, GRID_STEP
from src.runner.ModelRunner import ModelRunner

walls = [
    WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    WallInstance(id=2, length=3000, height=1800, weight=300, grid_step=GRID_STEP)
]

w = WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP)

runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")

res = runner.run(w)
for i in res.get('grid'):
    print(i)

print(res.get("instances"))
