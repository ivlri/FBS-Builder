from src.builder.structures import WallInstance, GRID_STEP
from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner
from src.optimizer.bonding_optimizer import BondingOptimizer


def test_chain_3_walls():
    """Test optimization on a chain of 3 walls."""
    print("=" * 60)
    print("Test: Chain of 3 walls")
    print("=" * 60)

    # Create 3 walls
    walls = [
        WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=4000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
    ]

    # Initialize components
    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    context_builder = ContextBuilder(grid_step=GRID_STEP)
    optimizer = BondingOptimizer(runner, context_builder)

    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize_chain(walls)

    print(f"\nResults:")
    print(f"  Total score: {result.total_score:.2f}")
    print(f"  RL calls: {result.num_rl_calls}")
    print(f"  Bonding assignments: {result.bonding_assignments}")

    print(f"\nPer-wall scores:")
    for wall_id, wall_result in result.wall_results.items():
        print(f"  Wall {wall_id}: score={wall_result.quality_score:.2f}, reward={wall_result.reward:.2f}")

    return result


def test_compare_with_default():
    """Compare optimizer result with default (no optimization)."""
    print("\n" + "=" * 60)
    print("Test: Compare optimizer vs default")
    print("=" * 60)

    walls = [
        WallInstance(id=1, length=3000, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=3500, height=1800, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=2500, height=1800, weight=300, grid_step=GRID_STEP),
    ]

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    context_builder = ContextBuilder(grid_step=GRID_STEP)

    # 1. Run with default bonding (no optimization)
    print("\n1. Running with DEFAULT bonding...")
    default_score = 0.0
    default_results = []

    for i, wall in enumerate(walls):
        context_data = {
            "walls": walls,
            "current_idx": i
        }
        result = runner.run(
            wall=wall,
            context_builder=context_builder,
            context_data=context_data,
        )
        score = _compute_quality(result)
        default_score += score
        default_results.append((wall.id, score, result.get("reward", 0)))

    print(f"  Default total score: {default_score:.2f}")
    for wall_id, score, reward in default_results:
        print(f"    Wall {wall_id}: score={score:.2f}, reward={reward:.2f}")

    # 2. Run with optimizer
    print("\n2. Running with OPTIMIZER...")
    optimizer = BondingOptimizer(runner, context_builder)
    opt_result = optimizer.optimize_chain(walls)

    print(f"  Optimized total score: {opt_result.total_score:.2f}")
    for wall_id, wall_result in opt_result.wall_results.items():
        print(f"    Wall {wall_id}: score={wall_result.quality_score:.2f}, reward={wall_result.reward:.2f}")

    # 3. Compare
    improvement = opt_result.total_score - default_score
    print(f"\n3. Comparison:")
    print(f"  Default:   {default_score:.2f}")
    print(f"  Optimized: {opt_result.total_score:.2f}")
    print(f"  Improvement: {improvement:+.2f} ({improvement/max(abs(default_score), 1)*100:+.1f}%)")
    print(f"  RL calls for optimization: {opt_result.num_rl_calls}")

    return default_score, opt_result.total_score


def _compute_quality(result) -> float:
    """Compute quality score (same as in optimizer)."""
    import numpy as np

    grid = result.get("grid")
    instances = result.get("instances", {})

    if grid is None or not instances:
        return 0.0

    total_cells = np.sum(grid > 0)
    if total_cells == 0:
        return 0.0

    monolith_cells = 0
    fbs_cells = 0
    seam_count = 0

    for inst in instances.values():
        type_id = inst.get("type_id", 0)
        length = inst.get("end", 0) - inst.get("start", 0)

        if type_id == 0:
            monolith_cells += length
        else:
            fbs_cells += length
            seam_count += 1

    fbs_ratio = fbs_cells / total_cells if total_cells > 0 else 0
    monolith_ratio = monolith_cells / total_cells if total_cells > 0 else 0

    score = (
        100 * fbs_ratio
        - 50 * monolith_ratio
        - 0.5 * seam_count
        + result.get("reward", 0) / 10
    )

    return score


def test_different_wall_lengths():
    """Test with walls of different lengths to see optimization impact."""
    print("\n" + "=" * 60)
    print("Test: Different wall lengths (where bonding matters more)")
    print("=" * 60)

    # Short walls where bonding choice has bigger relative impact
    walls = [
        WallInstance(id=1, length=2000, height=1200, weight=300, grid_step=GRID_STEP),
        WallInstance(id=2, length=1500, height=1200, weight=300, grid_step=GRID_STEP),
        WallInstance(id=3, length=2000, height=1200, weight=300, grid_step=GRID_STEP),
    ]

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    result = optimizer.optimize_chain(walls)

    print(f"\nResults for short walls:")
    print(f"  Total score: {result.total_score:.2f}")
    print(f"  RL calls: {result.num_rl_calls}")
    print(f"  Bonding: {result.bonding_assignments}")

    return result


if __name__ == "__main__":
    test_chain_3_walls()
    test_compare_with_default()
    test_different_wall_lengths()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
