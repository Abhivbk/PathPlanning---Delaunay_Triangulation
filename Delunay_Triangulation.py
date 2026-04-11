from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cone Object Definition
# -----------------------------
@dataclass
class Cone:
    x: float
    y: float
    cone_type: str  # e.g., "blue", "yellow", "orange", "unknown"

# -----------------------------
# Sample Cone List (Input Data)
# -----------------------------

def generate_sample_cones(num_points: int = 10, track_width: float = 2.0) -> List[Cone]:
    cones = []

    x_vals = np.linspace(0, 10, num_points)
    y_vals = np.sin(x_vals * 0.5) * 2

    for i in range(num_points):
        x = x_vals[i]
        y = y_vals[i]

        if i < num_points - 1:
            dx = x_vals[i+1] - x
            dy = y_vals[i+1] - y
        else:
            dx = x - x_vals[i-1]
            dy = y - y_vals[i-1]

        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length

        left_x = x + nx * track_width / 2
        left_y = y + ny * track_width / 2

        right_x = x - nx * track_width / 2
        right_y = y - ny * track_width / 2

        noise = 0.1
        left_x += np.random.uniform(-noise, noise)
        left_y += np.random.uniform(-noise, noise)

        right_x += np.random.uniform(-noise, noise)
        right_y += np.random.uniform(-noise, noise)

        cones.append(Cone(left_x, left_y, "blue"))
        cones.append(Cone(right_x, right_y, "yellow"))

    return cones

# -----------------------------
# Delaunay Triangulation Function
# -----------------------------
def delaunay_triangulation(cones: List[Cone]) -> List[Tuple[int, int, int]]:
    """
    Perform Delaunay triangulation on cone positions.

    Returns:
        List of triangles (each triangle = tuple of indices of cones)
    """
    from scipy.spatial import Delaunay

    # Convert cone list to numpy array of points
    points = np.array([[cone.x, cone.y] for cone in cones])

    # Perform triangulation
    tri = Delaunay(points)

    return tri.simplices  # indices of triangles

# Triangle ma with same color cones is useless
def filter_triangles(cones, triangles):
    filtered = []

    for t in triangles:
        c1, c2, c3 = cones[t[0]], cones[t[1]], cones[t[2]]

        types = {c1.cone_type, c2.cone_type, c3.cone_type}

        # Keep only triangles with more than 1 type
        if len(types) > 1:
            filtered.append(t)

    return filtered

# -----------------------------
# Utility: Print Triangles
# -----------------------------
def print_triangles(cones: List[Cone], triangles: List[Tuple[int, int, int]]):
    for t in triangles:
        c1, c2, c3 = cones[t[0]], cones[t[1]], cones[t[2]]
        print(f"Triangle:")
        print(f"  ({c1.x}, {c1.y}) [{c1.cone_type}]")
        print(f"  ({c2.x}, {c2.y}) [{c2.cone_type}]")
        print(f"  ({c3.x}, {c3.y}) [{c3.cone_type}]")
        print("-" * 30)
def plot_cones_and_triangles(cones: List[Cone], triangles: List[Tuple[int, int, int]], waypoints):
    # Plot cones
    for cone in cones:
        if cone.cone_type == "blue":
            color = "blue"
        elif cone.cone_type == "yellow":
            color = "yellow"
        elif cone.cone_type == "orange":
            color = "orange"
        else:
            color = "black"

        plt.scatter(cone.x, cone.y, color=color)
        plt.text(cone.x, cone.y, cone.cone_type, fontsize=8)

    # Plot triangles
    for t in triangles:
        c1, c2, c3 = cones[t[0]], cones[t[1]], cones[t[2]]

        # Triangle edges
        x_coords = [c1.x, c2.x, c3.x, c1.x]
        y_coords = [c1.y, c2.y, c3.y, c1.y]

        plt.plot(x_coords, y_coords, color = "black")

    # PLot Waypoints
    for point in waypoints:
        x, y = point
        plt.scatter(x, y, color="red")

    # Final touches
    plt.title("Delaunay Triangulation of Cones")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()

    plt.show()

def get_edges(triangles: List[Tuple[int, int, int]]):
    edges = set()
    for t in triangles:
        i,j,k = t
        edges.add((i,j))
        edges.add((j, k))
        edges.add((k, i))
    return edges
def useful_edges(cones, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    useful_edges = set()
    for edge in edges:
        c1, c2 = cones[edge[0]], cones[edge[1]]
        types = {c1.cone_type, c2.cone_type}
        if len(types) > 1:
            useful_edges.add(edge)
    return useful_edges

def waypoints(cones, useful_edges: List[Tuple[int, int]]):
    waypoints = []
    for edge in useful_edges:
        c1, c2 = cones[edge[0]], cones[edge[1]]
        x = (c1.x + c2.x)/2.0
        y = (c1.y + c2.y)/2.0
        waypoints.append((x, y))
    return waypoints
# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    # Step 1: Get cones
    cones = generate_sample_cones()

    # Step 2: Run Delaunay triangulation
    triangles = delaunay_triangulation(cones)
    triangles = filter_triangles(cones, triangles) # Filtering the triangles made with same color cones

    # Step 3: Output
    #print_triangles(cones, triangles)

    # Step 4: Display
    plot_cones_and_triangles(cones, triangles, waypoints(cones, useful_edges(cones, get_edges(triangles)) ))





if __name__ == "__main__":
    main()