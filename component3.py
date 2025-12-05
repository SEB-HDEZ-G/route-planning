# Imports --------------------------------------------------------------

import os
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPoint
from scipy.spatial import KDTree, Voronoi
from shapely.ops import clip_by_rect

# Matplotlib configuration for Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection 

import component1
import component2

# Functions ------------------------------------------------------------

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite 
    regions.
    Returns: list of regions (lists of points).
    """
    if vor.points.shape[0] == 0:
        return []

    if radius is None:
        pts = vor.points
        xmin = pts[:,0].min()
        xmax = pts[:,0].max()
        ymin = pts[:,1].min()
        ymax = pts[:,1].max()
        radius = math.hypot(xmax - xmin, ymax - ymin) * 2

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 not in vertices:
            new_regions.append([vor.vertices[v].tolist() for v in vertices])
            continue

        ridges = all_ridges.get(p1, [])
        points = []
        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                v_finite = v1 if v1 >= 0 else v2
                finite_vertex = vor.vertices[v_finite].tolist()
                tangent = vor.points[p2] - vor.points[p1]
                tangent = tangent / np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = finite_vertex + direction * radius
                points.append(tuple(finite_vertex))
                points.append(tuple(far_point.tolist()))
            else:
                points.append(tuple(vor.vertices[v1].tolist()))
                points.append(tuple(vor.vertices[v2].tolist()))

        if points:
            poly = MultiPoint(list(set(points))).convex_hull
            if poly.is_empty:
                new_regions.append([])
            else:
                new_regions.append(list(poly.exterior.coords))
        else:
             new_regions.append([])
             
    return new_regions

# Classes --------------------------------------------------------------

class HospitalRouting:
    def __init__(self, G_proj):
        self.G = G_proj
        
        # 1. Area Geometry
        # Calculate the map contour to clip Voronoi polygons later.
        # Simplify geometry to improve performance.
        node_points = [(data['x'], data['y']) for n, data in self.G.nodes(data=True)]
        self.area_polygon = MultiPoint(node_points).convex_hull.simplify(tolerance=50)

        # 2. KDTree for fast node lookup
        # Build a KDTree using node coordinates.
        self.node_ids = np.array(list(self.G.nodes()))
        self.coords = np.array([[self.G.nodes[n]['x'], self.G.nodes[n]['y']] for n in self.node_ids])
        self.node_kdtree = KDTree(self.coords)

        # 3. Prepare Graph for A* Search (Using Component 2)
        print("Preprocessing graph for searches...")
        # We use component2's helper to create a simple undirected weighted graph
        # and extract the coordinate dictionary required for the heuristic.
        self.Gu, self.coords_dict = component2.build_simple_undirected_weighted(self.G)
        
        # Initialization of state variables
        self.hospital_nodes = []
        self.hospital_coords = None
        self.hospital_names = []
        self.voronoi = []
        self.hospital_kdtree = None

    def find_hospitals(self):
        """
        Search for hospitals using OSM tags and strictly filter those 
        that fall within the graph's geometry area.
        """
        nodes_gdf, _ = ox.graph_to_gdfs(self.G, edges=True)
        nodes_latlon = nodes_gdf.to_crs(epsg=4326)
        # Use envelope for fast API query
        bbox = nodes_latlon.unary_union.envelope

        hospitals_raw = []
        
        # 1. Type Filter
        tags = {
            "amenity": ["hospital"], 
            "healthcare": ["hospital"]
        }
        
        try:
            hosp_gdf = ox.features_from_polygon(bbox, tags=tags)
        except Exception as e:
            print(f"Warning: {e}. Using fallback radial search...")
            center = nodes_latlon.unary_union.centroid
            hosp_gdf = ox.features_from_point((center.y, center.x), dist=5000, tags=tags)

        if hosp_gdf.empty:
            print("No hospitals found.")
            return

        import pyproj
        # Transformer: Lat/Lon -> Meters
        transformer = pyproj.Transformer.from_crs("EPSG:4326", self.G.graph['crs'], always_xy=True)
        
        for _, row in hosp_gdf.iterrows():
            if row.geometry is None: continue
            
            # Get coordinate (handle Point vs Polygon)
            geom_type = row.geometry.geom_type
            if geom_type == 'Point':
                pt = row.geometry.coords[0]
            else:
                pt = row.geometry.representative_point().coords[0]
            
            x, y = transformer.transform(pt[0], pt[1])
            
            # --- 2. Geographic Filter ---
            # Create a point and check if it is contained within the map area.
            # This removes hospitals from neighboring cities that shouldn't be routable.
            point_geom = Point(x, y)
            if not self.area_polygon.contains(point_geom):
                continue
            # ---------------------------

            name = row.get('name', 'Unnamed Hospital')
            if str(name) == 'nan': name = "Unnamed Hospital"
            hospitals_raw.append(((x, y), name))

        # Snap hospitals to the nearest graph nodes
        hosp_nodes_idx = []
        hosp_coords_final = []
        hosp_names_final = []

        for (x, y), name in hospitals_raw:
            dist, idx = self.node_kdtree.query((x, y))
            hosp_nodes_idx.append(self.node_ids[idx])
            hosp_coords_final.append((x, y))
            hosp_names_final.append(name)

        self.hospital_nodes = hosp_nodes_idx
        self.hospital_coords = np.array(hosp_coords_final)
        self.hospital_names = hosp_names_final

        if len(self.hospital_coords) > 0:
            self.hospital_kdtree = KDTree(self.hospital_coords)
            
        print(f"Hospitals within zone: {len(self.hospital_nodes)}")
        
    def build_voronoi(self):
        """Build Voronoi polygons and clip them to the map boundary."""
        if self.hospital_coords is None or len(self.hospital_coords) == 0:
            return

        vor = Voronoi(self.hospital_coords)
        regions = voronoi_finite_polygons_2d(vor)

        self.voronoi = []
        area_valid = self.area_polygon.buffer(0) # Topological cleanup

        for i, region in enumerate(regions):
            if not region: 
                self.voronoi.append((self.hospital_nodes[i], None, self.hospital_names[i]))
                continue
            
            poly = Polygon(region)
            try:
                # Clip: Intersection between infinite Voronoi and map area
                poly_clipped = poly.intersection(area_valid)
                self.voronoi.append((self.hospital_nodes[i], poly_clipped, self.hospital_names[i]))
            except Exception:
                self.voronoi.append((self.hospital_nodes[i], None, self.hospital_names[i]))
        
        print("Voronoi partition built.")

    def which_hospital_for_point(self, x, y):
        """Find the geometrically nearest hospital to a point X,Y."""
        dist, idx = self.hospital_kdtree.query((x, y))
        return int(idx), self.hospital_nodes[int(idx)], self.hospital_names[int(idx)], float(dist)

    def astar_route(self, source, target):
        """
        Calculates the shortest route using component2's A* implementation (SimpleAI).
        """
        start_t = time.perf_counter()
        
        # Call A* from component 2
        # Returns: path (list of nodes), cost (float)
        path, cost = component2.astar_search(self.Gu, source, target, self.coords_dict)
        
        elapsed = time.perf_counter() - start_t
        
        if cost == float('inf'):
             return None, cost, elapsed
             
        return path, cost, elapsed

    def snap_latlon_to_graph(self, lat, lon):
        """Converts user input Lat/Lon to the nearest graph node."""
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", self.G.graph['crs'], always_xy=True)
        x, y = transformer.transform(lon, lat)
        dist, idx = self.node_kdtree.query((x, y))
        return x, y, self.node_ids[idx], float(dist)

# GUI ------------------------------------------------------------------

class HospitalRoutingApp:
    def __init__(self, master, routing: HospitalRouting):
        self.master = master
        self.routing = routing
        master.title("Hospital Routing with Voronoi")

        # Configure Figure
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_axis_off()

        # Control Panel
        topframe = ttk.Frame(master)
        topframe.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(topframe, text="Lat:").pack(side=tk.LEFT)
        self.lat_entry = ttk.Entry(topframe, width=12)
        self.lat_entry.pack(side=tk.LEFT, padx=(0,10))
        # Default value (Zapopan center)
        self.lat_entry.insert(0, "20.72") 

        ttk.Label(topframe, text="Lon:").pack(side=tk.LEFT)
        self.lon_entry = ttk.Entry(topframe, width=12)
        self.lon_entry.pack(side=tk.LEFT, padx=(0,10))
        # Default value
        self.lon_entry.insert(0, "-103.39") 

        ttk.Button(topframe, text="Calculate Route", command=self.on_find_route).pack(side=tk.LEFT)
        ttk.Button(topframe, text="View Voronoi Regions", command=self.plot_voronoi).pack(side=tk.LEFT, padx=5)

        # Canvas for the map
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # GRAPHICS CACHE: Prepare lines for LineCollection (100x faster)
        print("Preparing graphics...")
        self.lines_data = []
        for u, v, data in self.routing.G.edges(data=True):
            x1 = self.routing.G.nodes[u]['x']
            y1 = self.routing.G.nodes[u]['y']
            x2 = self.routing.G.nodes[v]['x']
            y2 = self.routing.G.nodes[v]['y']
            self.lines_data.append([(x1, y1), (x2, y2)])

        self.plot_base_map()

    def plot_base_map(self):
        self.ax.clear()
        self.ax.set_axis_off()
        
        # Draw all streets at once
        lc = LineCollection(self.lines_data, colors='black', linewidths=0.5, alpha=0.5)
        self.ax.add_collection(lc)
        self.ax.autoscale()
        self.canvas.draw_idle()

    def plot_voronoi(self):
        self.ax.clear()
        self.ax.set_axis_off()

        # 1. Faint base map (background)
        lc = LineCollection(self.lines_data, colors='gray', linewidths=0.3, alpha=0.2)
        self.ax.add_collection(lc)

        # 2. Voronoi Polygons (colored regions)
        for node, poly, name in self.routing.voronoi:
            if poly and not poly.is_empty:
                xs, ys = poly.exterior.xy
                self.ax.fill(xs, ys, alpha=0.4, ec='white')

        # 3. Hospital Points and Names
        coords = self.routing.hospital_coords
        names = self.routing.hospital_names
        
        if len(coords) > 0:
            # Draw all red points first
            self.ax.scatter(coords[:,0], coords[:,1], c='red', marker='P', s=60, edgecolors='white', zorder=10)
            
            # Draw texts one by one
            for (x, y), name in zip(coords, names):
                # Shorten name if too long (first 15 chars)
                short_name = name[:15] + "..." if len(name) > 15 else name
                
                self.ax.annotate(
                    short_name, 
                    (x, y),              # Point coordinate
                    xytext=(0, -8),      # Offset: 0 horizontal, -8 vertical (down)
                    textcoords='offset points', 
                    ha='center',         # Horizontal alignment: Center
                    va='top',            # Vertical alignment: Top (text hangs from point)
                    fontsize=6,          # Small font to fit
                    fontweight='bold',
                    zorder=11            # Appear on top
                )

        self.ax.autoscale()
        self.canvas.draw_idle()

    def on_find_route(self):
        try:
            lat = float(self.lat_entry.get())
            lon = float(self.lon_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Lat/Lon must be numbers.")
            return

        # 1. Find start node
        x, y, start_node, dstart = self.routing.snap_latlon_to_graph(lat, lon)
        
        # 2. Find nearest hospital
        idx, hosp_node, hosp_name, dist_hosp = self.routing.which_hospital_for_point(x, y)
        
        # 3. Calculate route (Using Component 2 A*)
        print(f"Calculating A* path from {start_node} to {hosp_node}...")
        path, cost, t_elapsed = self.routing.astar_route(start_node, hosp_node)

        if not path:
            messagebox.showinfo("Error", "No valid path found (or graph disconnected).")
            return

        # 4. Draw Result
        self.ax.clear()
        self.ax.set_axis_off()
        
        # Light gray background
        lc = LineCollection(self.lines_data, colors='#DDDDDD', linewidths=0.5)
        self.ax.add_collection(lc)

        # Draw Route (bright blue)
        # We need to look up coords in the graph
        path_points = [(self.routing.G.nodes[n]['x'], self.routing.G.nodes[n]['y']) for n in path]
        px, py = zip(*path_points)
        self.ax.plot(px, py, color='blue', linewidth=2.5, alpha=0.9, zorder=5)

        # Start and End
        self.ax.scatter(x, y, c='green', s=80, label='Your Location', zorder=6, edgecolors='black')
        hx, hy = self.routing.hospital_coords[idx]
        self.ax.scatter(hx, hy, c='red', marker='P', s=100, label=f'Hospital: {hosp_name}', zorder=6, edgecolors='white')
        
        self.ax.legend(loc='upper right')
        self.ax.autoscale()
        self.canvas.draw_idle()
        
        msg = f"Route to: {hosp_name}\nApprox Distance: {cost:.0f} meters\nSimpleAI A* Time: {t_elapsed:.4f} sec"
        print(msg)
        messagebox.showinfo("Route Found", msg)


# Main Execution -------------------------------------------------------

def main():
    # Location Configuration
    campus_name = 'Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México'
    filename = "graph_zapopan.graphml" 

    # Load or Download using Component 1 logic
    if os.path.exists(filename):
        print(f"Loading local map '{filename}'...")
        G_proj = ox.load_graphml(filename)
    else:
        print("Local file not found. Using Component 1 to download/load...")
        # Use component 1 loader
        G, G_proj = component1.load_campus_graph(campus_name, buffer_meters=6000, network_type="drive")
        print("Saving graph to disk...")
        ox.save_graphml(G_proj, filename)

    print(f"Map ready: {len(G_proj)} nodes.")
    
    # Initialize Logic
    routing = HospitalRouting(G_proj)
    routing.find_hospitals()
    routing.build_voronoi()

    # Initialize App
    root = tk.Tk()
    # Maximize window if possible
    try:
        root.state('zoomed')
    except:
        pass
    app = HospitalRoutingApp(root, routing)
    root.mainloop()

if __name__ == "__main__":
    main()