import matplotlib

matplotlib.use("Agg")

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.builder.structures import GRID_STEP, WallInstance
from src.planner.overhang import EdgeType, OverhangAnalyzer
from src.planner.wall_planner import WallPlanner
from src.solver.solver_pipeline import (
    PipelineResult,
    SolverPipeline,
    visualize_pipeline,
)

Point = Tuple[float, float]


# ============================================================
# VISUALIZATION HELPERS
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
        return "  (empty)"
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
                lines.append(f"  L{layer}r0 | {text}")
            if row1:
                merged = _merge_consecutive(row1)
                text = ", ".join(f"{b['type_id']}({b['length_mm']}mm)" for b in merged)
                lines.append(f"  L{layer}r1 | {text}")
    return "\n".join(lines)


# ============================================================
# HELPER FUNCTION
# ============================================================


def run_pipeline_test(
    x_start: List[float],
    y_start: List[float],
    x_end: List[float],
    y_end: List[float],
    wall_ids: List[int],
    test_name: str,
    height: int = 1800,
    weight: int = 300,
) -> PipelineResult:
    """
    Universal runner for pipeline tests.
    Points -> WallPlanner -> SolverPipeline -> Visualization.
    """
    # 1. WallPlanner
    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=x_start,
        y_start=y_start,
        x_end=x_end,
        y_end=y_end,
        wall_ids=wall_ids,
        heights=[height] * len(wall_ids),
        weights=[weight] * len(wall_ids),
    )
    planner.process()

    # 2. Get walls in traversal order
    walls = planner.get_wall_instances()

    # 3. Run pipeline
    pipeline = SolverPipeline(grid_step=GRID_STEP)
    pipeline.set_wall_graph(planner.processed_graph)

    # Setup wall_nodes for overhang constraints
    for item in planner.traversal:
        pipeline.wall_nodes[item.wall_id] = (item.start_point, item.end_point)

    result = pipeline.solve_chain(walls)

    # 4. Visualize
    print(f"\n{'=' * 70}")
    print(f"{test_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Walls: {len(walls)}")
    print(f"Traversal order: {[w.id for w in walls]}")
    stats = planner.get_stats()
    print(
        f"Graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges, "
        f"{stats['connected_components']} components"
    )
    print(visualize_pipeline(result, walls, GRID_STEP))
    print(f"Total: {result.total_stats}")

    return result


# ============================================================
# TESTS
# ============================================================


def test_simple_rectangle():
    """Simple rectangle: 4 walls forming closed contour."""
    result = run_pipeline_test(
        x_start=[0, 4000, 4000, 0],
        y_start=[0, 0, 3000, 3000],
        x_end=[4000, 4000, 0, 0],
        y_end=[0, 3000, 3000, 0],
        wall_ids=[1, 2, 3, 4],
        test_name="Simple Rectangle (4000x3000)",
    )

    assert len(result.wall_results) == 4
    assert result.total_stats["total_fbs"] > 0


def test_l_shape():
    """L-shaped building: 6 walls."""
    result = run_pipeline_test(
        x_start=[0, 4000, 4000, 2000, 2000, 0],
        y_start=[0, 0, 2000, 2000, 4000, 4000],
        x_end=[4000, 4000, 2000, 2000, 0, 0],
        y_end=[0, 2000, 2000, 4000, 4000, 0],
        wall_ids=[1, 2, 3, 4, 5, 6],
        test_name="L-Shape Building",
    )

    assert len(result.wall_results) == 6
    assert result.total_stats["total_fbs"] > 0


def test_door_opening():
    """Building with door opening (gap 800-1100mm)."""
    # Wall 1: 0,0 -> 1000,0 (before door)
    # Gap: 1000,0 -> 2000,0 (door, 1000mm)
    # Wall 2: 2000,0 -> 5000,0 (after door)
    # Wall 3-5: rest of building
    result = run_pipeline_test(
        x_start=[0, 2000, 5000, 5000, 0],
        y_start=[0, 0, 0, 3000, 3000],
        x_end=[1000, 5000, 5000, 0, 0],
        y_end=[0, 0, 3000, 3000, 0],
        wall_ids=[1, 2, 3, 4, 5],
        test_name="Building with Door Opening",
    )

    assert len(result.wall_results) == 5


def test_t_junction():
    """T-junction: 2 horizontal walls + 1 vertical branch."""
    result = run_pipeline_test(
        x_start=[0, 2000, 2000],
        y_start=[2000, 2000, 2000],
        x_end=[2000, 4000, 2000],
        y_end=[2000, 2000, 0],
        wall_ids=[1, 2, 3],
        test_name="T-Junction",
    )

    assert len(result.wall_results) == 3
    assert result.total_stats["total_fbs"] > 0


def test_free_edge():
    """Single wall with free edges on both sides."""
    result = run_pipeline_test(
        x_start=[0],
        y_start=[0],
        x_end=[3000],
        y_end=[0],
        wall_ids=[1],
        test_name="Single Wall (Free Edges)",
    )

    assert len(result.wall_results) == 1
    assert result.total_stats["total_fbs"] > 0

    # Check overhang constraints via analyzer
    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=[0], y_start=[0], x_end=[3000], y_end=[0], wall_ids=[1]
    )
    planner.process()
    analyzer = OverhangAnalyzer(planner.processed_graph)
    constraints = analyzer.analyze_wall((0.0, 0.0), (3000.0, 0.0), "1")

    assert constraints.left_edge.edge_type == EdgeType.FREE_EDGE
    assert constraints.right_edge.edge_type == EdgeType.FREE_EDGE
    print(
        f"  Overhang: L={constraints.left_edge.max_overhang_mm}mm, "
        f"R={constraints.right_edge.max_overhang_mm}mm"
    )


def test_wall_with_opening():
    """Wall with window opening (1200x1200mm centered at 3000mm)."""
    from src.builder.structures import Opening

    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=[0],
        y_start=[0],
        x_end=[6000],
        y_end=[0],
        wall_ids=[1],
        heights=[1800],
        weights=[300],
    )
    planner.process()

    walls = planner.get_wall_instances()

    # Create opening: window 1200x1200mm, center at x=3000, y=900
    opening = Opening(center_x=3000, center_y=900, width=1200, height=1200)

    pipeline = SolverPipeline(grid_step=GRID_STEP)
    pipeline.set_wall_graph(planner.processed_graph)

    for item in planner.traversal:
        pipeline.wall_nodes[item.wall_id] = (item.start_point, item.end_point)

    # Solve with opening
    result = pipeline.solve_chain(walls, openings_map={1: [opening]})

    print(f"\n{'=' * 70}")
    print("WALL WITH OPENING TEST")
    print(f"{'=' * 70}")
    print(f"Opening: {opening}")
    print(visualize_pipeline(result, walls, GRID_STEP))
    print(f"Total: {result.total_stats}")

    assert len(result.wall_results) == 1
    assert result.total_stats["total_fbs"] > 0

    # Verify opening is respected (blocked cells in grid)
    wall_result = result.wall_results[1]
    grid = wall_result.grid

    # Opening spans cells 120-180 (3000-600 to 3000+600 = 2400-3600mm)
    # and rows 1-4 (900-600 to 900+600 = 300-1500mm → rows 1,2,3,4)
    # Check that opening area is blocked (-1)
    opening_cells = (3000 - 600) // GRID_STEP  # 120
    opening_rows = (900 - 600) // 300  # row 1
    assert grid[opening_rows, opening_cells] == -1, "Opening should be blocked"
    print(f"  Opening correctly blocked at row {opening_rows}, cell {opening_cells}")


def test_real_data():
    """Real data from Revit: 34 walls."""
    from src.planner.planer import TEST_DATA

    result = run_pipeline_test(
        x_start=[float(x) for x in TEST_DATA["x_start"]],
        y_start=[float(y) for y in TEST_DATA["y_start"]],
        x_end=[float(x) for x in TEST_DATA["x_end"]],
        y_end=[float(y) for y in TEST_DATA["y_end"]],
        wall_ids=TEST_DATA["wall_id"],
        test_name="Real Building Data (34 walls)",
    )

    assert len(result.wall_results) == 34
    assert result.total_stats["total_fbs"] > 0

    # Summary by wall
    print("\n  Per-wall summary:")
    for wall_id, wall_result in result.wall_results.items():
        stats = wall_result.stats
        oh_l = wall_result.left_overhang_mm
        oh_r = wall_result.right_overhang_mm
        print(
            f"    Wall {wall_id}: {stats['fbs_count']} FBS, "
            f"{stats['monolith_cells'] * GRID_STEP}mm mono, "
            f"overhang L={oh_l}mm R={oh_r}mm"
        )


def test_pipeline_with_overhang():
    """Test pipeline with overhang constraints (T-junction scenario)."""
    # T-junction: wall 3 connects at interior joint
    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=[0, 2000, 2000],
        y_start=[2000, 2000, 2000],
        x_end=[2000, 4000, 2000],
        y_end=[2000, 2000, 0],
        wall_ids=[1, 2, 3],
    )
    planner.process()

    walls = planner.get_wall_instances()

    pipeline = SolverPipeline(grid_step=GRID_STEP)
    pipeline.set_wall_graph(planner.processed_graph)
    for item in planner.traversal:
        pipeline.wall_nodes[item.wall_id] = (item.start_point, item.end_point)

    # Check overhang constraints
    oh1 = pipeline.get_overhang_constraints(1)
    oh2 = pipeline.get_overhang_constraints(2)
    oh3 = pipeline.get_overhang_constraints(3)

    print(f"\n{'=' * 70}")
    print("T-JUNCTION OVERHANG CONSTRAINTS")
    print(f"{'=' * 70}")
    print(f"  Wall 1: L={oh1[0]}mm R={oh1[1]}mm")
    print(f"  Wall 2: L={oh2[0]}mm R={oh2[1]}mm")
    print(f"  Wall 3: L={oh3[0]}mm R={oh3[1]}mm")

    # Check that overhang system is working (not all zeros)
    all_constraints = [oh1, oh2, oh3]
    has_interior = any(c[0] == 500 or c[1] == 500 for c in all_constraints)
    assert has_interior, "T-junction should have at least one interior joint (500mm)"

    result = pipeline.solve_chain(walls)
    assert result.total_stats["total_fbs"] > 0
    print(f"\n  Result: {result.total_stats}")


def test_overhang_visualization():
    """Test that overhang symbols < > appear in visualization."""
    from src.solver.solver_pipeline import SolverResult, visualize_wall

    # Create fake result with overhang blocks
    num_rows = 6
    num_cells = 100
    grid = np.zeros((num_rows, num_cells), dtype=int)
    grid[0:2, 10:70] = 1  # Block 1 inside wall
    grid[2:4, 0:60] = 2  # Block 2 inside wall

    # Instance with left overhang (start_cell = -10, so 10 cells overhang)
    # Instance with right overhang (end_cell = 110, so 10 cells overhang)
    instances = {
        1: {"row": 0, "start_cell": 10, "end_cell": 70, "h_rows": 2, "type_id": 2},
        2: {
            "row": 2,
            "start_cell": -10,
            "end_cell": 60,
            "h_rows": 2,
            "type_id": 3,
        },  # Left overhang
        3: {
            "row": 4,
            "start_cell": 40,
            "end_cell": 110,
            "h_rows": 2,
            "type_id": 4,
        },  # Right overhang
    }

    result = SolverResult(
        wall_id=99,
        grid=grid,
        instances=instances,
        stats={"fbs_count": 3, "monolith_cells": 0},
        blocked_mask=None,
        left_overhang_mm=200,
        right_overhang_mm=200,
    )

    wall = WallInstance(
        id=99, length=2000, height=1800, weight=300, grid_step=GRID_STEP
    )

    viz = visualize_wall(result, wall, GRID_STEP)

    print(f"\n{'=' * 70}")
    print("OVERHANG VISUALIZATION TEST")
    print(f"{'=' * 70}")
    print(viz)

    # Check that < and > symbols appear
    assert "<" in viz, "Left overhang symbol '<' should appear in visualization"
    assert ">" in viz, "Right overhang symbol '>' should appear in visualization"
    print("\n  [OK] Overhang symbols < > correctly displayed")


def test_adjusted_length():
    """Test wall length adjustment for neighbors."""
    # L-shape: two walls meeting at corner (both 300mm thick)
    planner = WallPlanner(grid_step=GRID_STEP)
    planner.add_walls_from_coords(
        x_start=[0, 4000],
        y_start=[0, 0],
        x_end=[4000, 4000],
        y_end=[0, 3000],
        wall_ids=[1, 2],
        weights=[300, 300],
    )
    planner.process()
    walls = planner.get_wall_instances()

    # Wall 1: 4000mm + 150mm (neighbor at END) = 4150mm -> 4140mm (rounded to 20)
    # Wall 2: 3000mm + 150mm (neighbor at START) = 3150mm -> 3140mm (rounded to 20)
    print(f"\n  Wall 1: base=4000mm, adjusted={walls[0].length}mm")
    print(f"  Wall 2: base=3000mm, adjusted={walls[1].length}mm")

    assert walls[0].length == 4140, f"Wall 1: expected 4140, got {walls[0].length}"
    assert walls[1].length == 3140, f"Wall 2: expected 3140, got {walls[1].length}"

    # Test wall with neighbors on BOTH ends (chain: wall2 - wall1 - wall3)
    planner2 = WallPlanner(grid_step=GRID_STEP)
    planner2.add_walls_from_coords(
        x_start=[0, -2000, 4000],
        y_start=[0, 0, 0],
        x_end=[4000, 0, 6000],
        y_end=[0, 0, 0],
        wall_ids=[1, 2, 3],
        weights=[300, 300, 300],
    )
    planner2.process()
    walls2 = planner2.get_wall_instances()

    # Find wall 1 in results (middle wall with neighbors at both ends)
    wall1 = next(w for w in walls2 if w.id == 1)
    # Wall 1: 4000mm + 150mm (start) + 150mm (end) = 4300mm
    print(f"  Wall 1 (two neighbors): base=4000mm, adjusted={wall1.length}mm")
    assert wall1.length == 4300, f"Wall 1: expected 4300, got {wall1.length}"

    # Test single wall (no neighbors)
    planner3 = WallPlanner(grid_step=GRID_STEP)
    planner3.add_walls_from_coords(
        x_start=[0],
        y_start=[0],
        x_end=[3000],
        y_end=[0],
        wall_ids=[1],
    )
    planner3.process()
    walls3 = planner3.get_wall_instances()
    print(f"  Single wall: base=3000mm, adjusted={walls3[0].length}mm (no change)")
    assert walls3[0].length == 3000, (
        f"Single wall: expected 3000, got {walls3[0].length}"
    )


# ============================================================
# RUNNER
# ============================================================

TESTS = {
    "rect": ("Simple Rectangle", test_simple_rectangle),
    "lshape": ("L-Shape", test_l_shape),
    "door": ("Door Opening", test_door_opening),
    "tjunc": ("T-Junction", test_t_junction),
    "free": ("Free Edge", test_free_edge),
    "real": ("Real Data (34 walls)", test_real_data),
    "overhang": ("Pipeline with Overhang", test_pipeline_with_overhang),
    "opening": ("Wall with Opening", test_wall_with_opening),
    "viz": ("Overhang Visualization", test_overhang_visualization),
    "adjlen": ("Adjusted Length", test_adjusted_length),
}


def run_tests(test_keys: List[str]) -> bool:
    """Run specified tests."""
    print("=" * 70)
    print("FULL PIPELINE TESTS: Points -> WallPlanner -> SolverPipeline")
    print("=" * 70)

    passed = 0
    failed = 0

    for key in test_keys:
        if key not in TESTS:
            print(f"\n  [ERR] Unknown test: {key}")
            failed += 1
            continue

        name, test_func = TESTS[key]
        print(f"\n{'=' * 70}")
        print(f"TEST: {name}")
        print(f"{'=' * 70}")
        try:
            test_func()
            print(f"\n  [OK] PASSED")
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  [ERR] ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline tests: Points -> WallPlanner -> SolverPipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "test",
        nargs="?",
        default="all",
        choices=["all"] + list(TESTS.keys()),
        help="""Test to run:
                all      - run all tests (default)
                rect     - Simple Rectangle (4 walls)
                lshape   - L-Shape (6 walls)
                door     - Door Opening (5 walls)
                tjunc    - T-Junction (3 walls)
                free     - Free Edge (1 wall)
                real     - Real Data (34 walls)
                overhang - Pipeline with Overhang""",
    )

    args = parser.parse_args()

    if args.test == "all":
        test_keys = list(TESTS.keys())
    else:
        test_keys = [args.test]

    success = run_tests(test_keys)
    exit(0 if success else 1)

    # Все тесты
    # python -m src.tests.overhang_test

    # # Отдельные тесты
    # python -m src.tests.overhang_test rect      # Simple Rectangle
    # python -m src.tests.overhang_test lshape    # L-Shape
    # python -m src.tests.overhang_test door      # Door Opening
    # python -m src.tests.overhang_test tjunc     # T-Junction
    # python -m src.tests.overhang_test free      # Free Edge
    # python -m src.tests.overhang_test real      # Real Data (34 walls)
    # python -m src.tests.overhang_test overhang  # Pipeline with Overhang
