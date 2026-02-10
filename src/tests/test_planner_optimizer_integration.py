"""
Integration test: WallPlanner + BondingOptimizer.

Tests the full pipeline from raw wall coordinates to optimized layouts.
"""

from src.planner.wall_planner import WallPlanner, WallData
from src.optimizer.bonding_optimizer import BondingOptimizer
from src.contextbuilder.contextbuilder import ContextBuilder
from src.runner.ModelRunner import ModelRunner
from src.builder.structures import GRID_STEP


# Sample data from planer.py (subset for testing)
SAMPLE_X_START = [14975, 14975, 8500, 10750, 10375]
SAMPLE_Y_START = [25, 3375, 6625, 7250, 7250]
SAMPLE_X_END = [7700, 10375, 8500, 10750, 10375]
SAMPLE_Y_END = [25, 3375, 11625, 12000, 3375]
SAMPLE_WALL_IDS = [2662653, 2662654, 2662660, 2662663, 2662665]


def test_simple_chain():
    """Test with manually created chain of 4 walls."""
    print("=" * 60)
    print("Test: Simple chain (4 walls)")
    print("=" * 60)

    # Create simple chain: W1 -- W2 -- W3 -- W4
    walls = [
        WallData(wall_id=1, x_start=0, y_start=0, x_end=3000, y_end=0, height=1800, weight=300),
        WallData(wall_id=2, x_start=3000, y_start=0, x_end=6000, y_end=0, height=1800, weight=300),
        WallData(wall_id=3, x_start=6000, y_start=0, x_end=9000, y_end=0, height=1800, weight=300),
        WallData(wall_id=4, x_start=9000, y_start=0, x_end=12000, y_end=0, height=1800, weight=300),
    ]

    # Plan
    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls(walls)
    planner.process()

    print(f"\nPlanner stats: {planner.get_stats()}")
    print(f"Traversal order: {planner.get_traversal_order()}")
    print(f"Adjacency: {planner.get_adjacency()}")

    # Optimize
    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    wall_instances = planner.get_wall_instances()
    adjacency = planner.get_adjacency()

    print(f"\nWall instances: {[(w.id, w.length) for w in wall_instances]}")

    result = optimizer.optimize(wall_instances, adjacency)

    print(f"\nOptimization results:")
    print(f"  Total score: {result.total_score:.2f}")
    print(f"  RL calls: {result.num_rl_calls}")
    print(f"  Bonding: {result.bonding_assignments}")

    for wall_id, wall_result in result.wall_results.items():
        print(f"  Wall {wall_id}: score={wall_result.quality_score:.2f}")

    return result


def test_l_shape():
    """Test L-shaped configuration (2 walls at 90 degrees)."""
    print("\n" + "=" * 60)
    print("Test: L-shape (2 walls at 90°)")
    print("=" * 60)

    walls = [
        WallData(wall_id=1, x_start=0, y_start=0, x_end=3000, y_end=0, height=1800, weight=300),
        WallData(wall_id=2, x_start=3000, y_start=0, x_end=3000, y_end=3000, height=1800, weight=300),
    ]

    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls(walls)
    planner.process()

    print(f"Traversal: {planner.get_traversal_order()}")
    print(f"Adjacency: {planner.get_adjacency()}")

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    result = optimizer.optimize(
        planner.get_wall_instances(),
        planner.get_adjacency()
    )

    print(f"\nTotal score: {result.total_score:.2f}")
    print(f"RL calls: {result.num_rl_calls}")
    print(f"Bonding: {result.bonding_assignments}")

    return result


def test_t_junction():
    """Test T-junction (3 walls meeting at one point)."""
    print("\n" + "=" * 60)
    print("Test: T-junction (3 walls)")
    print("=" * 60)

    # T-shape:
    #      W2
    #       |
    # W1 ---+--- W3

    walls = [
        WallData(wall_id=1, x_start=0, y_start=0, x_end=3000, y_end=0, height=1800, weight=300),
        WallData(wall_id=2, x_start=3000, y_start=0, x_end=3000, y_end=3000, height=1800, weight=300),
        WallData(wall_id=3, x_start=3000, y_start=0, x_end=6000, y_end=0, height=1800, weight=300),
    ]

    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls(walls)
    planner.process()

    stats = planner.get_stats()
    print(f"Stats: {stats}")
    print(f"Traversal: {planner.get_traversal_order()}")
    print(f"Adjacency (chain): {planner.get_adjacency()}")
    print(f"Adjacency (graph): {planner.get_adjacency_from_graph()}")

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    # Use graph-based adjacency for T-junction
    result = optimizer.optimize(
        planner.get_wall_instances(),
        planner.get_adjacency_from_graph()
    )

    print(f"\nTotal score: {result.total_score:.2f}")
    print(f"RL calls: {result.num_rl_calls}")
    print(f"Bonding: {result.bonding_assignments}")

    return result


def test_real_data_subset():
    """Test with subset of real data from planer.py."""
    print("\n" + "=" * 60)
    print("Test: Real data subset (5 walls)")
    print("=" * 60)

    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=SAMPLE_X_START,
        y_start=SAMPLE_Y_START,
        x_end=SAMPLE_X_END,
        y_end=SAMPLE_Y_END,
        wall_ids=SAMPLE_WALL_IDS,
        heights=[1800] * 5,
        weights=[300] * 5,
    )
    planner.process()

    stats = planner.get_stats()
    print(f"Stats: {stats}")
    print(f"Traversal: {planner.get_traversal_order()}")

    wall_instances = planner.get_wall_instances()
    print(f"Wall lengths: {[(w.id, w.length) for w in wall_instances]}")

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    result = optimizer.optimize(
        wall_instances,
        planner.get_adjacency_from_graph()
    )

    print(f"\nTotal score: {result.total_score:.2f}")
    print(f"RL calls: {result.num_rl_calls}")
    print(f"Bonding: {result.bonding_assignments}")

    for wall_id, wall_result in sorted(result.wall_results.items()):
        print(f"  Wall {wall_id}: score={wall_result.quality_score:.2f}, reward={wall_result.reward:.2f}")

    return result


def test_compare_chain_vs_graph_adjacency():
    """Compare results using chain adjacency vs graph adjacency."""
    print("\n" + "=" * 60)
    print("Test: Chain vs Graph adjacency comparison")
    print("=" * 60)

    # T-shape where adjacency method matters
    walls = [
        WallData(wall_id=1, x_start=0, y_start=0, x_end=3000, y_end=0, height=1800, weight=300),
        WallData(wall_id=2, x_start=3000, y_start=0, x_end=3000, y_end=3000, height=1800, weight=300),
        WallData(wall_id=3, x_start=3000, y_start=0, x_end=6000, y_end=0, height=1800, weight=300),
    ]

    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls(walls)
    planner.process()

    runner = ModelRunner(model_path="src/builder/data/ppo_fbs_builder")
    optimizer = BondingOptimizer(runner, ContextBuilder(grid_step=GRID_STEP))

    # Chain adjacency (simpler, may miss connections)
    chain_adj = planner.get_adjacency()
    print(f"Chain adjacency: {chain_adj}")

    result_chain = optimizer.optimize(
        planner.get_wall_instances(),
        chain_adj
    )

    # Graph adjacency (accurate for T-junctions)
    graph_adj = planner.get_adjacency_from_graph()
    print(f"Graph adjacency: {graph_adj}")

    result_graph = optimizer.optimize(
        planner.get_wall_instances(),
        graph_adj
    )

    print(f"\nChain adjacency score: {result_chain.total_score:.2f} (RL calls: {result_chain.num_rl_calls})")
    print(f"Graph adjacency score: {result_graph.total_score:.2f} (RL calls: {result_graph.num_rl_calls})")

    return result_chain, result_graph


if __name__ == "__main__":
    test_simple_chain()
    test_l_shape()
    test_t_junction()
    test_real_data_subset()
    test_compare_chain_vs_graph_adjacency()

    print("\n" + "=" * 60)
    print("All integration tests completed!")
    print("=" * 60)
