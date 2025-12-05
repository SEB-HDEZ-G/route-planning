# Imports --------------------------------------------------------------

import osmnx as ox
import pyproj
import numpy as np
from scipy.spatial import KDTree
import time

# Functions ------------------------------------------------------------

def load_campus_graph(campus_name='Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México', 
                      buffer_meters=1000, 
                      network_type="walk"):
    """
    Load and return the campus graph from OSMnx.
    Returns: (G, G_proj) tuple of original and projected graphs
    """
    try:
        gdf = ox.geocode_to_gdf(campus_name)
        if not gdf.empty:
            print("Found campus polygon:")
            print(gdf)
            # Use polygon to build the graph
            polygon = gdf.geometry.iloc[0]
            
            # Buffer the polygon to be able to expand zone (distance in meters)
            # Project to UTM for accurate distance buffering
            from shapely.ops import transform
            import pyproj
            
            # Create projection transformers
            wgs84 = pyproj.CRS('EPSG:4326')
            # Use UTM zone for Guadalajara area
            utm = pyproj.CRS('EPSG:32613')  # UTM Zone 13N
            
            project_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
            project_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
            
            # Transform, buffer, transform back
            polygon_utm = transform(project_to_utm, polygon)
            buffered_polygon_utm = polygon_utm.buffer(buffer_meters)  # Meters expansion
            polygon_buffered = transform(project_to_wgs84, buffered_polygon_utm)
            
            G = ox.graph_from_polygon(polygon_buffered, network_type=network_type)
        else:
            print("geocode_to_gdf returned empty result. Using fallback instead")
            raise ValueError
    except Exception as e:
        print(f"Could not get polygon via geocode - fallback to buffered point. Error: {e}")

        # Fallback: buffered point
        center = (20.7348, -103.4540)  # lat, lon
        buffer_m = 500  # e.g. 500 metres - adjust as needed
        G = ox.graph_from_point(center, dist=buffer_m, network_type=network_type)

    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    
    # Project the graph
    G_proj = ox.project_graph(G)
    
    return G, G_proj

def exhaustiveSearch(pt):
    min_dist = float('inf')
    min_idx = -1
    
    for i in range(len(coords)):
        # Calculate Euclidean distance manually
        dx = coords[i][0] - pt[0]
        dy = coords[i][1] - pt[1]
        dist = (dx*dx + dy*dy) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_idx

# Main -----------------------------------------------------------------

# Module-level variables
G = None
G_proj = None
coords = None
node_ids = None
tree = None

if __name__ == "__main__":

    # FIND ZONE IN MAP AND PLOT AS GRAPH ===============================
    campus_name = 'Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México'
    
    G, G_proj = load_campus_graph(campus_name, buffer_meters=1000, network_type="walk")
    ox.plot_graph(G, node_size=5, figsize=(8, 8))

    # Get projections used by OSMnx
    projector = pyproj.Transformer.from_crs("epsg:4326", G_proj.graph['crs'], always_xy=True)

    def project_point(lon, lat):
        X, Y = projector.transform(lon, lat)
        return X, Y

    # BUILD KD-TREE WITH GRAPH VERTICES ================================

    # Extract node coordinates in projected space
    nodes = G_proj.nodes(data=True)
    coords = np.array([(data['x'], data['y']) for _, data in nodes])  # shape: (N, 2)
    node_ids = np.array([n for n, _ in nodes])  # Keep IDs for later

    t0 = time.time()
    tree = KDTree(coords)
    t_build = time.time() - t0

    print("\nTiming units ")
    print("-" * 40)
    print("Build time: ms = milliseconds")
    print("Search time: μs = microseconds")

    print(f"\nKD-Tree build time: {t_build*1000:.3f} ms")

    # KD-TREE SEARCH FOR 20 COORDINATES ================================

    # Extract 20 actual points from graph nodes
    graph_nodes = list(G.nodes(data=True))
    # Take every Nth node to get a good distribution across the graph
    step = max(1, len(graph_nodes) // 20)
    sample_nodes = graph_nodes[::step][:20]

    # Extract (lat, lon) from the sampled nodes
    test_points = [(data['y'], data['x']) for _, data in sample_nodes]  # (lat, lon)

    print(f"\nUsing {len(test_points)} real points from the graph:")
    for i, (lat, lon) in enumerate(test_points):
        print(f"  Point {i+1}: ({lat:.6f}, {lon:.6f})")

    projected_points = [project_point(lon, lat) for lat, lon in test_points]

    # Run multiple iterations for more accurate timing
    ITERATIONS = 100
    kd_times = []
    closest_nodes_kd = []

    for pt in projected_points:
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            dist, idx = tree.query(pt)
        t_kd = (time.perf_counter() - t0) / ITERATIONS
        kd_times.append(t_kd)
        
        closest_nodes_kd.append(node_ids[idx])

    print(f"\nKD-Tree average search time: {np.mean(kd_times)*1_000_000:.3f} μs")

    # COMPARISON WITH EXHAUSTIVE SEARCH ================================

    exh_times = []
    closest_nodes_exh = []

    for pt in projected_points:
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            idx = exhaustiveSearch(pt)
        t_exh = (time.perf_counter() - t0) / ITERATIONS
        exh_times.append(t_exh)
        
        closest_nodes_exh.append(node_ids[idx])

    print(f"Exhaustive average search time: {np.mean(exh_times)*1_000_000:.3f} μs")
    print(f"Speedup: KD-Tree is {np.mean(exh_times)/np.mean(kd_times):.2f}x faster\n")

    for i in range(len(test_points)):
        print(f"Point {i+1}: KD={closest_nodes_kd[i]}, EXH={closest_nodes_exh[i]}")