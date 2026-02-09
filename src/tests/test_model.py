import numpy as np
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from builder.fbs_builder import FBSBuilderEnv, WallInstance, Opening, BLOCK_TYPES
from src.builder.structures import WallInstance, Opening, GRID_STEP

def mask_fn(env):
    """Extract action mask from potentially wrapped env."""
    while hasattr(env, 'env'):
        if hasattr(env, 'get_action_mask'):
            return env.get_action_mask()
        env = env.env
    return env.get_action_mask()

# Must match training parameters!
MIN_LENGTH = 1200
MAX_LENGTH = 6000
MIN_HEIGHT = 1200
MAX_HEIGHT = 3000

WALLS = {
    "small": WallInstance(id=0, length=3000, height=1200, weight=300, grid_step=GRID_STEP),
    "medium": WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    "large": WallInstance(id=2, length=4000, height=2400, weight=300, grid_step=GRID_STEP),
    "wide": WallInstance(id=3, length=5000, height=3000, weight=300, grid_step=GRID_STEP),
    "tall": WallInstance(id=4, length=3000, height=3000, weight=300, grid_step=GRID_STEP),
}

WALLS_WITH_OPENINGS = {
    "medium_door": {
        "wall": WallInstance(id=10, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        "openings": [Opening(center_x=1500, center_y=600, width=400, height=600)],
    },
    "large_window": {
        "wall": WallInstance(id=11, length=4000, height=2400, weight=300, grid_step=GRID_STEP),
        "openings": [Opening(center_x=2000, center_y=1500, width=500, height=400)],
    },
    "wide_two_openings": {
        "wall": WallInstance(id=12, length=5000, height=3000, weight=300, grid_step=GRID_STEP),
        "openings": [
            Opening(center_x=1500, center_y=900, width=400, height=600),
            Opening(center_x=3500, center_y=1500, width=500, height=400),
        ],
    },
}


def get_base_env(vec_env):
    """Get base FBSBuilderEnv from wrapped vec env"""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


def compute_block_statistics(grid_human, instances, num_cells, num_rows):
    """
    Compute block type statistics from final grid.
    Returns dict with block counts and percentages
    """
    stats = {
        "total_cells": 0,
        "monolith_300_cells": 0,
        "monolith_600_cells": 0,
        "fbs_cells": 0,
        "blocks_by_type": {},
    }

    # Count blocks by type from instances
    for inst_id, meta in instances.items():
        type_id = meta["type_id"]
        block_type = next((bt for bt in BLOCK_TYPES if bt.id == type_id), None)

        if block_type:
            if type_id not in stats["blocks_by_type"]:
                stats["blocks_by_type"][type_id] = {
                    "name": block_type.name,
                    "count": 0,
                    "cells": 0,
                }

            stats["blocks_by_type"][type_id]["count"] += 1
            block_cells = meta["end"] - meta["start"]
            stats["blocks_by_type"][type_id]["cells"] += block_cells

            # Categorize
            if type_id == 0:  # Monolith
                stats["monolith_300_cells"] += block_cells
            # elif type_id == 1:  # Monolith 600
            #     stats["monolith_600_cells"] += block_cells
            else:
                stats["fbs_cells"] += block_cells

    # Total non-empty cells
    filled_cells = np.sum(grid_human[:num_rows, :num_cells] != 0)
    stats["total_cells"] = filled_cells

    # Percentages
    if filled_cells > 0:
        stats["monolith_pct"] = (stats["monolith_300_cells"] + stats["monolith_600_cells"]) / filled_cells * 100
        stats["fbs_pct"] = stats["fbs_cells"] / filled_cells * 100
    else:
        stats["monolith_pct"] = 0
        stats["fbs_pct"] = 0

    return stats


def print_block_statistics(stats):
    """Pretty print block statistics."""
    print("\n" + "="*60)
    print("BLOCK STATISTICS")
    print("="*60)
    print(f"Total filled cells: {stats['total_cells']}")
    print(f"FBS blocks:         {stats['fbs_cells']:5d} cells ({stats['fbs_pct']:5.1f}%)")
    print(f"Monolith:           {stats['monolith_300_cells'] + stats['monolith_600_cells']:5d} cells ({stats['monolith_pct']:5.1f}%)")

    if stats["blocks_by_type"]:
        print("\nBlocks by type:")
        for type_id in sorted(stats["blocks_by_type"].keys()):
            info = stats["blocks_by_type"][type_id]
            print(f"  {info['name']:15s}: {info['count']:3d} blocks, {info['cells']:5d} cells")

    # Warnings
    if stats["monolith_pct"] > 10:
        print(f"\n[WARNING] Monolith usage {stats['monolith_pct']:.1f}% > 10% - policy may be degraded")

    print("="*60)


def make_vec_env(wall, openings=None, vec_normalize_path="src/builder/data/vec_normalize.pkl"):
    """Create vectorized env with VecNormalize loaded from training."""
    def make_env():
        env = FBSBuilderEnv(
            wall_instance=wall,
            openings=openings,
            render_mode="terminal_human",
            max_steps=500,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            min_height=MIN_HEIGHT,
            max_height=MAX_HEIGHT,
            grid_step=GRID_STEP,
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_env])

    # Load VecNormalize
    if os.path.exists(vec_normalize_path):
        try:
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            print(f"[INFO] Loaded VecNormalize from {vec_normalize_path}")
        except AssertionError as e:
            print(f"[WARNING] VecNormalize shape mismatch: {e}")
            print("[INFO] Creating new VecNormalize (model was retrained)")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, norm_obs_keys=["grid", "blocked_mask"])
    else:
        print("[INFO] VecNormalize not found, creating new (inference without normalization)")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, norm_obs_keys=["grid", "blocked_mask"])

    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def validation(wall_type: str = "medium", model_path: str = "src/builder/data/ppo_fbs_builder"):
    """
    Test model on a specific wall type using deterministic inference.
    """
    if wall_type not in WALLS:
        print(f"Unknown wall type: {wall_type}")
        print(f"Available: {list(WALLS.keys())}")
        return

    wall = WALLS[wall_type]

    print(f"\n{'='*60}")
    print(f"Testing wall: {wall_type}")
    print(f"Size: {wall.length}mm x {wall.height}mm")
    print(f"Grid: {wall.num_cells} cells x {wall.num_layers} layers")
    print(f"{'='*60}\n")

    vec_env = make_vec_env(wall)
    model = MaskablePPO.load(model_path)

    # Deterministic inference
    obs = vec_env.reset()
    done = False
    steps = 0
    base_env = get_base_env(vec_env)

    while not done:
        masks = base_env.get_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, rewards, dones, infos = vec_env.step(action)
        steps += 1
        done = dones[0]
        info = infos[0]

    # Get terminal state
    reward = info.get('total_reward', 0.0)
    grid = info.get('terminal_grid')
    instances = info.get('terminal_instances')
    reason = info.get('reason', '?')

    # Restore for rendering
    base_env.grid_human = grid
    base_env.inst = instances
    base_env.render()

    # Print statistics
    stats = compute_block_statistics(grid, instances, wall.num_cells, wall.num_rows)
    print_block_statistics(stats)

    print(f"\nResult: reward={reward:.1f}, steps={steps}, reason={reason}")

    # Quality checks
    if reason != "all_rows_completed":
        print(f"[FAIL] Wall not completed (reason: {reason})")
    elif stats["monolith_pct"] > 15:
        print(f"[WARNING] Excessive monolith usage ({stats['monolith_pct']:.1f}%)")
    elif reward < 100:
        print(f"[WARNING] Low reward ({reward:.1f})")
    else:
        print("[OK] Wall construction successful")

    return reward


def validation_with_openings(config_name: str = "medium_door",
                             model_path: str = "src/builder/data/ppo_fbs_builder"):
    """Test model on a wall with openings using deterministic inference"""
    if config_name not in WALLS_WITH_OPENINGS:
        print(f"Unknown config: {config_name}")
        print(f"Available: {list(WALLS_WITH_OPENINGS.keys())}")
        return

    cfg = WALLS_WITH_OPENINGS[config_name]
    wall = cfg["wall"]
    openings = cfg["openings"]

    print(f"\n{'='*60}")
    print(f"Testing wall with openings: {config_name}")
    print(f"Size: {wall.length}mm x {wall.height}mm")
    print(f"Grid: {wall.num_cells} cells x {wall.num_layers} layers")
    print(f"Openings: {len(openings)}")

    for i, op in enumerate(openings):
        print(f"    [{i}] center=({op.center_x}, {op.center_y})mm, size={op.width}x{op.height}mm")

    print(f"{'='*60}\n")

    vec_env = make_vec_env(wall, openings=openings)
    model = MaskablePPO.load(model_path)

    # Deterministic inference
    obs = vec_env.reset()
    done = False
    steps = 0
    base_env = get_base_env(vec_env)

    while not done:
        masks = base_env.get_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, rewards, dones, infos = vec_env.step(action)
        steps += 1
        done = dones[0]
        info = infos[0]

    # Get terminal state
    reward = info.get('total_reward', 0.0)
    grid = info.get('terminal_grid')
    instances = info.get('terminal_instances')
    reason = info.get('reason', '?')

    # Restore for rendering
    base_env.grid_human = grid
    base_env.inst = instances
    base_env.render()

    # Statistics
    stats = compute_block_statistics(grid, instances, wall.num_cells, wall.num_rows)
    print_block_statistics(stats)

    print(f"\nResult: reward={reward:.1f}, steps={steps}, reason={reason}")

    # Quality checks
    if reason != "all_rows_completed":
        print(f"[FAIL] Wall not completed (reason: {reason})")
    elif stats["monolith_pct"] > 15:
        print(f"[WARNING] Excessive monolith usage ({stats['monolith_pct']:.1f}%)")
    elif reward < 100:
        print(f"[WARNING] Low reward ({reward:.1f})")
    else:
        print("[OK] Wall construction successful")

    return reward


def test_all(model_path: str = "src/builder/data/ppo_fbs_builder"):
    """Test model on all wall types (clean + with openings)"""
    results = {}

    for wall_type in WALLS:
        reward = validation(wall_type, model_path)
        results[wall_type] = reward

    for config_name in WALLS_WITH_OPENINGS:
        reward = validation_with_openings(config_name, model_path)
        results[config_name] = reward

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, reward in results.items():
        status = "OK" if reward >= 100 else "LOW"
        print(f"  {name:20}: {reward:>8.1f}  [{status}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FBS model")
    all_choices = list(WALLS.keys()) + list(WALLS_WITH_OPENINGS.keys()) + ["all"]

    parser.add_argument("--wall", "-w", default="medium", choices=all_choices, help="Wall type to test")
    parser.add_argument("--model", "-m", default="src/builder/data/ppo_fbs_builder", help="Path to model")

    args = parser.parse_args()

    if args.wall == "all":
        test_all(args.model)
    elif args.wall in WALLS_WITH_OPENINGS:
        validation_with_openings(args.wall, args.model)
    else:
        validation(args.wall, args.model)
