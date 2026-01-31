import numpy as np
from sb3_contrib import MaskablePPO
from FBSByilder import FBSBuilderEnv, WallInstance, Opening

GRID_STEP = 20
WALLS = {
    "small": WallInstance(id=0, length=3000, height=1200, weight=300, grid_step=GRID_STEP),
    "medium": WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    "large": WallInstance(id=2, length=4000, height=2400, weight=300, grid_step=GRID_STEP),
    "wide": WallInstance(id=3, length=5000, height=3000, weight=300, grid_step=GRID_STEP),
    "tall": WallInstance(id=4, length=3000, height=3000, weight=300, grid_step=GRID_STEP),
}

CONFIDENCE_THRESHOLD = 0.0
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


def best_of_n_predict(model, env, n=5):
    """
    Run N stochastic episodes return the best result by total reward
    """
    best_reward = -float('inf')
    best_grid = None
    best_instances = None
    best_steps = 0
    best_info = {}

    for i in range(n):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            masks = env.get_action_mask()
            action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        if env.total_reward > best_reward:
            best_reward = env.total_reward
            best_grid = env.grid_human.copy()
            best_instances = dict(env.instances)
            best_steps = steps
            best_info = dict(info)

    return best_grid, best_instances, best_reward, best_steps, best_info


def validation(wall_type: str = "medium", model_path: str = "ppo_fbs_builder", n_attempts: int = 5):
    """
    Test model on a specific wall type using best-of-N stochastic inference
    """
    if wall_type not in WALLS:
        print(f"Unknown wall type: {wall_type}")
        print(f"Available: {list(WALLS.keys())}")
        return

    wall = WALLS[wall_type]

    print(f"\n{'='*60}")
    print(f"Testing wall: {wall_type} (best-of-{n_attempts})")
    print(f"Size: {wall.length}mm x {wall.height}mm")
    print(f"Grid: {wall.num_cells} cells x {wall.num_layers} layers")
    print(f"{'='*60}\n")

    env = FBSBuilderEnv(wall_instance=wall, render_mode="terminal_human", max_steps=500)
    model = MaskablePPO.load(model_path)

    #baseline
    obs, _ = env.reset()
    done = False
    det_steps = 0
    while not done:
        masks = env.get_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, info = env.step(action)
        det_steps += 1
        done = terminated or truncated
    det_reward = env.total_reward
    det_info = info

    print(f"Deterministic: reward={det_reward:.1f}, steps={det_steps}, "f"reason={det_info.get('reason', '?')}")

    #Best of N stochastic
    grid, instances, best_reward, best_steps, best_info = best_of_n_predict(
        model, env, n=n_attempts
    )

    #Restore best result
    env.grid_human = grid
    env.inst = instances
    env.render()

    print(f"\nBest of {n_attempts}: reward={best_reward:.1f}, steps={best_steps}, reason={best_info.get('reason', '?')}")
    print(f"Improvement: {best_reward - det_reward:+.1f}")

    #Confidence scoring
    if best_reward < CONFIDENCE_THRESHOLD:
        print(f"\n[WARNING] Best reward {best_reward:.1f} < threshold {CONFIDENCE_THRESHOLD}. Model may need more training for this config.")

    return best_reward


def validation_with_openings(config_name: str = "medium_door",
                             model_path: str = "ppo_fbs_builder",
                             n_attempts: int = 5):
    """Test model on a wall with openings using best-of-N stochastic inference"""
    if config_name not in WALLS_WITH_OPENINGS:
        print(f"Unknown config: {config_name}")
        print(f"Available: {list(WALLS_WITH_OPENINGS.keys())}")
        return

    cfg = WALLS_WITH_OPENINGS[config_name]
    wall = cfg["wall"]
    openings = cfg["openings"]

    print(f"\n{'='*60}")
    print(f"Testing wall with openings: {config_name} (best-of-{n_attempts})")
    print(f"Size: {wall.length_mm}mm x {wall.height_mm}mm")
    print(f"Grid: {wall.num_cells} cells x {wall.num_layers} layers")
    print(f"Openings: {len(openings)}")

    for i, op in enumerate(openings):
        print(f"    [{i}] center=({op.center_x_mm}, {op.center_y_mm})mm, size={op.width_mm}x{op.height_mm}mm")

    print(f"{'='*60}\n")

    env = FBSBuilderEnv(wall_instance=wall, openings=openings,
                        render_mode="terminal_human", max_steps=500)
    model = MaskablePPO.load(model_path)

    #baseline
    obs, _ = env.reset()
    done = False
    det_steps = 0
    while not done:
        masks = env.get_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, info = env.step(action)
        det_steps += 1
        done = terminated or truncated
    det_reward = env.total_reward
    det_info = info

    print(f"Deterministic: reward={det_reward:.1f}, steps={det_steps}, reason={det_info.get('reason', '?')}")

    #Stochastic
    grid, instances, best_reward, best_steps, best_info = best_of_n_predict(
        model, env, n=n_attempts
    )

    env.grid_human = grid
    env.inst = instances
    env.render()

    print(f"\nBest-of-{n_attempts}: reward={best_reward:.1f}, steps={best_steps}, reason={best_info.get('reason', '?')}")
    print(f"Improvement: {best_reward - det_reward:+.1f}")

    if best_reward < CONFIDENCE_THRESHOLD:
        print(f"\n[WARNING] Best reward {best_reward:.1f} < threshold {CONFIDENCE_THRESHOLD}. Model may need more training for this config.")

    return best_reward


def test_all(model_path: str = "ppo_fbs_builder", n_attempts: int = 5):
    """Test model on all wall types (clean + with openings)"""
    results = {}
    for wall_type in WALLS:
        reward = validation(wall_type, model_path, n_attempts=n_attempts)
        results[wall_type] = reward

    for config_name in WALLS_WITH_OPENINGS:
        reward = validation_with_openings(config_name, model_path, n_attempts=n_attempts)
        results[config_name] = reward

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, reward in results.items():
        status = "OK" if reward >= CONFIDENCE_THRESHOLD else "LOW"
        print(f"  {name:20}: {reward:>8.1f}  [{status}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FBS model")
    all_choices = list(WALLS.keys()) + list(WALLS_WITH_OPENINGS.keys()) + ["all"]

    parser.add_argument("--wall", "-w", default="medium", choices=all_choices, help="Wall type to test")

    parser.add_argument("--model", "-m", default="ppo_fbs_builder", help="Path to model")

    parser.add_argument("--attempts", "-n", type=int, default=5, help="Number of stochastic attempts (best-of-N)")

    args = parser.parse_args()

    if args.wall == "all":
        test_all(args.model, n_attempts=args.attempts)
    elif args.wall in WALLS_WITH_OPENINGS:
        validation_with_openings(args.wall, args.model, n_attempts=args.attempts)
    else:
        validation(args.wall, args.model, n_attempts=args.attempts)
