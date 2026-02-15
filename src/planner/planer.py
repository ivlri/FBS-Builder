import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List, Dict, Any, Optional, Set
from collections import defaultdict
import numpy as np
from collections import deque

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

def normalize_point(p: Point) -> Point:
    """Нормализует точку для избежания float ошибок."""
    return (round(p[0], 6), round(p[1], 6))

def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    def ccw(A: Point, B: Point, C: Point) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-9: 
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
        return normalize_point((x1 + t*(x2-x1), y1 + t*(y2-y1)))
    
    return None

def _on_segment(pt: Point, a: Point, b: Point, eps=1e-3) -> bool:
    cross = (pt[1]-a[1])*(b[0]-a[0]) - (pt[0]-a[0])*(b[1]-a[1])
    dot = (pt[0]-a[0])*(b[0]-a[0]) + (pt[1]-a[1])*(b[1]-a[1])
    sqr_len = (b[0]-a[0])**2 + (b[1]-a[1])**2

    return abs(cross) < eps * sqr_len and 0 < dot < sqr_len

def is_proper_intersection(p: Point, seg1: Segment, seg2: Segment) -> bool:
    return _on_segment(p, *seg1) and + _on_segment(p, *seg2)

def point_to_segment_distance(p: Point, a: Point, b: Point) -> Tuple[float, Point]:
    """Distance from point p to segment (a, b) and projection point."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return ((px - ax)**2 + (py - ay)**2)**0.5, a
    
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    proj = (ax + t * dx, ay + t * dy)
    dist = ((px - proj[0])**2 + (py - proj[1])**2)**0.5
    
    return dist, normalize_point(proj)


def split_edge(G: nx.Graph, 
               u: Point, 
               v: Point, 
               p: Point, 
               wall_id: str, 
               suffix: str) -> bool:
    u_norm, v_norm, p_norm = normalize_point(u), normalize_point(v), normalize_point(p)

    if not G.has_edge(u_norm, v_norm):
        return False

    data = G[u_norm][v_norm]
    wall_id_orig = data['wall_id']
    orig = data.get('original_wall_id', wall_id_orig)

    G.remove_edge(u_norm, v_norm)
    G.add_edge(u_norm, p_norm, 
               wall_id=f"{wall_id_orig}{suffix}_1", 
               original_wall_id=orig)
    G.add_edge(p_norm, v_norm, 
               wall_id=f"{wall_id_orig}{suffix}_2", 
               original_wall_id=orig)
    return True

def process_intersections(G: nx.Graph, verbose: bool = False) -> nx.Graph:
    G_proc = G.copy()

    for _ in range(5):
        current_edges = list(G_proc.edges(data=True))
        intersections_found = False

        for i, (u1,v1,d1) in enumerate(current_edges):
            for j, (u2,v2,d2) in enumerate(current_edges[i+1:], i+1):
                p = line_intersection(u1, v1, u2, v2)
                if p and is_proper_intersection(p, (u1,v1), (u2,v2)):
                    if verbose:
                        print(f"Intersection: {u1}-{v1} x {u2}-{v2} = {p}")
                    split_edge(G_proc, u1, v1, p, d1['wall_id'], '_A')
                    split_edge(G_proc, u2, v2, p, d2['wall_id'], '_B')
                    intersections_found = True
                    break

            if intersections_found:
                break

        if not intersections_found:
            break

    return G_proc

def process_t_joints(G: nx.Graph, tolerance: float = 50.0, verbose: bool = False) -> nx.Graph:
    """Find nodes that lie on edges from a different component (T-joints) and split."""
    G_proc = G.copy()

    for iteration in range(50):
        found = False
        components = list(nx.connected_components(G_proc))
        if len(components) <= 1:
            break

        node_to_comp = {}
        for i, comp in enumerate(components):
            for n in comp:
                node_to_comp[n] = i

        for p in list(G_proc.nodes()):
            best_dist = tolerance
            best_edge = None

            p_comp = node_to_comp[p]
            for u, v, data in list(G_proc.edges(data=True)):
                if node_to_comp[u] == p_comp:
                    continue

                if u == p or v == p:
                    continue

                dist, proj = point_to_segment_distance(p, u, v)
                if dist < best_dist:
                    d_to_u = ((proj[0]-u[0])**2 + (proj[1]-u[1])**2)**0.5
                    d_to_v = ((proj[0]-v[0])**2 + (proj[1]-v[1])**2)**0.5

                    if d_to_u > tolerance/2 and d_to_v > tolerance/2:
                        best_dist = dist
                        best_edge = (u, v, data)

            if best_edge is not None:
                u, v, data = best_edge
                if verbose:
                    print(f"T-joint: node {p} -> edge {u}-{v} (dist={best_dist:.1f}mm)")
                split_edge(G_proc, u, v, p, data['wall_id'], '_T')
                found = True
                break

        if not found:
            break

    return G_proc


def choose_start_node(G: nx.Graph) -> Point:
    """Pick best start node: prefer highdegree corners, then degree 1 endpoints."""
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("Empty graph")

    return max(nodes, key=lambda n: (G.degree(n), n))


def _bfs_to_nearest_unvisited(G: nx.Graph, start: Point,
                               visited_originals: Set[str]) -> Optional[List[Point]]:
    """BFS from start to nearest node adjacent to an unvisited original_wall_id edge."""
    queue = deque([(start, [start])])
    seen = {start}

    while queue:
        node, path = queue.popleft()

        for nbr in G.neighbors(node):
            orig = G[node][nbr].get('original_wall_id', G[node][nbr]['wall_id'])

            if orig not in visited_originals:
                return path + [nbr]
            
            if nbr not in seen:
                seen.add(nbr)
                queue.append((nbr, path + [nbr]))
    return None


def traverse_walls(G: nx.Graph, start: Point) -> List[Tuple[Tuple[Point, Point], str]]:
    """Edge-based traversal tracking original wall id. Nodes can be revisited."""
    visited_originals: Set[str] = set()
    traversal = []
    current = normalize_point(start)

    # Collect all unique wall ids
    all_originals = set()
    for u, v, d in G.edges(data=True):
        all_originals.add(d.get('original_wall_id', d['wall_id']))

    while len(visited_originals) < len(all_originals):
        # Find adjacent edge with original wall id
        best_edge = None
        for nbr in G.neighbors(current):
            d = G[current][nbr]
            orig = d.get('original_wall_id', d['wall_id'])
            if orig not in visited_originals:
                best_edge = (current, nbr, d)
                break

        if best_edge is not None:
            u, v, d = best_edge
            orig = d.get('original_wall_id', d['wall_id'])
            visited_originals.add(orig)
            # Walk along connected sub edges of the same wall starting from current
            walked = _walk_original_wall(G, current, orig)
            for eu, ev in walked:
                edge = tuple(sorted([eu, ev]))
                traversal.append((edge, orig))
            if walked:
                current = walked[-1][1]  # End of the last sub edge
            else:
                edge = tuple(sorted([u, v]))
                traversal.append((edge, orig))
                current = v
        else:
            # All neighbors visited - BFS to nearest unvisited
            path = _bfs_to_nearest_unvisited(G, current, visited_originals)
            if path is None:
                # Disconnected component - jump to any node with unvisited wall
                jumped = False
                for u, v, d in G.edges(data=True):
                    orig = d.get('original_wall_id', d['wall_id'])
                    if orig not in visited_originals:
                        current = u
                        jumped = True
                        break
                if not jumped:
                    break
            else:
                current = path[-1]


    return traversal


def _walk_original_wall(G: nx.Graph, start: Point,
                        original_id: str) -> List[Tuple[Point, Point]]:
    """Walk along all sub-edges of the same original_wall_id starting from start."""
    result = []
    current = start
    visited_edges = set()

    while True:
        found = False
        for nbr in G.neighbors(current):
            d = G[current][nbr]
            orig = d.get('original_wall_id', d['wall_id'])
            edge_key = tuple(sorted([current, nbr]))

            if orig == original_id and edge_key not in visited_edges:
                visited_edges.add(edge_key)
                result.append((current, nbr))
                current = nbr
                found = True
                break

        if not found:
            break

    return result

# Test data for visualization
TEST_DATA = {
    "x_start": [14975,14975,8500,10750,10375,25,14975,14975,13250,14975,13245,6700,14975,
               14975,16845,18475,525,10750,18475,525,8125,8500,8500,8500,12000,14975,18525,
               14975,21850,21850,18525,18525,14975,14975],
    "y_start": [25,3375,6625,7250,7250,13400,25,20850,12000,12000,13875,25,8150,
               6650,8150,12000,25,7250,6650,13400,11625,11625,13375,13375,13875,
               13875,19100,13875,11875,12875,13609,20100,22410,21120],
    "x_end": [7700,10375,8500,10750,10375,25,14975,25,10750,14975,9650,525,16015,18475,21850,
             14250,525,9720,18475,25,8125,8125,8125,8500,12000,14975,14975,14245,21850,21850,
             18525,21850,11771,14975],
    "y_end": [25,3375,11625,12000,3375,20850,8500,20850,12000,9460,13875,25,8150,
             6650,8150,12000,13400,7250,12000,13400,13375,11625,13375,15580,20850,
             21004,19100,13875,6496,20100,20100,20100,22410,22410],
    "wall_id": [2662653,2662654,2662660,2662663,2662665,2662676,2662684,2662693,
               2662695,2662696,2662748,2662750,2662754,2662756,2848644,3095721,3153305,
               3153524,3153680,3153785,3154057,3154296,3154367,3154565,3154788,3155014,
               3155106,3155212,3155770,3155910,3155911,3155912,3156179,3156238],
}


def create_test_graph() -> nx.Graph:
    """Create graph from test data."""
    x_start = TEST_DATA["x_start"]
    y_start = TEST_DATA["y_start"]
    x_end = TEST_DATA["x_end"]
    y_end = TEST_DATA["y_end"]
    wall_id = TEST_DATA["wall_id"]

    zip_start = [(float(x), float(y)) for x, y in zip(x_start, y_start)]
    zip_end = [(float(x), float(y)) for x, y in zip(x_end, y_end)]

    G = nx.Graph()
    G.add_edges_from([(normalize_point(st),
                       normalize_point(end),
                       {"wall_id": str(wid),
                        "original_wall_id": str(wid)
                        })
                      for st, end, wid in zip(zip_start, zip_end, wall_id)])
    return G


def visualize_graph(G_original: nx.Graph, G_merged: nx.Graph,
                    traversal: List, start_node: Point):
    """Visualize original and merged graphs with traversal."""
    pos = {node: node for node in set(G_original.nodes()) | set(G_merged.nodes())}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Original graph
    components = list(nx.connected_components(G_original))
    colors = plt.cm.tab20(np.linspace(0,1,len(components)))
    for i, comp in enumerate(components):
        nx.draw(G_original.subgraph(comp), pos, ax=ax1, node_size=30,
                node_color=colors[i], edge_color=colors[i], alpha=0.7)
    ax1.set_title(f"Original graph ({len(components)} components)")

    # Merged graph + traversal
    nx.draw(G_merged, pos, ax=ax2, node_size=40, node_color='lightblue',
            edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G_merged, pos, ax=ax2, nodelist=[start_node],
                           node_color='red', node_size=200)

    covered_ids = set(wall for _, wall in traversal)
    for i, ((u,v),_) in enumerate(traversal):
        nx.draw_networkx_edges(G_merged, pos, ax=ax2, edgelist=[(u,v)],
                               edge_color='orange', width=3)
        nx.draw_networkx_edge_labels(G_merged, pos, ax=ax2,
                                    edge_labels={(u,v):str(i+1)}, font_size=7)

    n_comp = nx.number_connected_components(G_merged)
    ax2.set_title(f"Merged ({n_comp} comp.) + traversal ({len(covered_ids)} walls)")

    for ax in [ax1, ax2]:
        ax.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create and process graph
    G_original = create_test_graph()
    print("Original: components =", nx.number_connected_components(G_original),
          "edges =", G_original.number_of_edges())

    G_cross = process_intersections(G_original, verbose=True)
    print("After intersections: components =", nx.number_connected_components(G_cross),
          "edges =", G_cross.number_of_edges())

    G_merged = process_t_joints(G_cross, verbose=True)
    print("After T-joints: components =", nx.number_connected_components(G_merged),
          "edges =", G_merged.number_of_edges())

    start_node = choose_start_node(G_merged)
    print("Start:", start_node, "degree =", G_merged.degree(start_node))

    traversal = traverse_walls(G_merged, start_node)

    all_original_ids = set()
    for u, v, d in G_merged.edges(data=True):
        all_original_ids.add(d.get('original_wall_id', d['wall_id']))

    covered_ids = set(wall for _, wall in traversal)

    print(f"\nTraversal: {len(traversal)} entries, "
          f"covered {len(covered_ids)}/{len(all_original_ids)} walls")
    for i, ((u,v), wall) in enumerate(traversal, 1):
        print(f"{i}: {u}->{v} original_wall={wall}")

    missing = all_original_ids - covered_ids
    if missing:
        print(f"\nNot covered: {missing}")
    else:
        print("\nAll walls covered!")

    # Visualize
    visualize_graph(G_original, G_merged, traversal, start_node)
