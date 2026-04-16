import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional


# Constants
LEFT_TYPES = {"blue", "small_orange", "large_orange"}
RIGHT_TYPES = {"yellow", "small_orange", "large_orange"}
CENTERLINE_LEFT_TYPES = {"blue"}
CENTERLINE_RIGHT_TYPES = {"yellow"}


@dataclass
class Cone:
    """Represents a cone with position and type."""
    x: float
    y: float
    cone_type: str  # "blue" | "yellow" | "small_orange" | "large_orange"


@dataclass
class Triangle:
    """Represents a triangle from Delaunay triangulation with three cone indices."""
    indices: Tuple[int, int, int]

    def __post_init__(self):
        self.indices = tuple(sorted(self.indices))

    def get_cone_types(self, cones: List[Cone]) -> List[str]:
        return [cones[i].cone_type for i in self.indices]

    def has_mixed_types(self, cones: List[Cone]) -> bool:
        types = set(self.get_cone_types(cones))
        return bool(types & LEFT_TYPES and types & RIGHT_TYPES)

    def max_edge_length(self, cones: List[Cone]) -> float:
        p = [cones[i] for i in self.indices]
        d1 = np.hypot(p[0].x - p[1].x, p[0].y - p[1].y)
        d2 = np.hypot(p[1].x - p[2].x, p[1].y - p[2].y)
        d3 = np.hypot(p[2].x - p[0].x, p[2].y - p[0].y)
        return max(d1, d2, d3)


class Track:
    """
    Represents the FSA track with cones, triangles, centerline, and racing line.

    Attributes:
        track_width (float): Nominal track width in meters.
        cones (List[Cone]): List of cones on the track.
        triangles (List[Triangle]): Filtered Delaunay triangles.
        centerline (List[Tuple[float, float]]): Raw centerline points.
        smoothed_centerline (List[Tuple[float, float]]): Smoothed centerline (Lap 1).
        racing_line (List[Tuple[float, float]]): Min-curvature racing line (Lap 2+).
    """

    def __init__(self, track_width: float = 3.5):
        self.track_width = track_width
        self.cones: List[Cone] = []
        self.triangles: List[Triangle] = []
        self.centerline: List[Tuple[float, float]] = []
        self.smoothed_centerline: List[Tuple[float, float]] = []
        self.racing_line: List[Tuple[float, float]] = []

    def build_cones(self):
        """Build the track cones: rectangle with rounded hairpin corners."""
        hw = self.track_width / 2
        R = 7.0
        SL = 50.0
        sp = 3.0

        for x in np.arange(sp, SL, sp):
            self.cones.append(Cone(x, hw, "blue"))
            self.cones.append(Cone(x, -hw, "yellow"))

        cx_r, cy_r = SL, R
        for ang in np.linspace(np.radians(-90), np.radians(90), 37):
            ox, oy = np.cos(ang), np.sin(ang)
            px, py = cx_r + R * ox, cy_r + R * oy
            self.cones.append(Cone(px - ox * hw, py - oy * hw, "blue"))
            self.cones.append(Cone(px + ox * hw, py + oy * hw, "yellow"))

        for x in np.arange(SL - sp, 0, -sp):
            self.cones.append(Cone(x, 2 * R - hw, "blue"))
            self.cones.append(Cone(x, 2 * R + hw, "yellow"))

        cx_l, cy_l = 0.0, R
        for ang in np.linspace(np.radians(90), np.radians(270), 37):
            ox, oy = np.cos(ang), np.sin(ang)
            px, py = cx_l + R * ox, cy_l + R * oy
            self.cones.append(Cone(px - ox * hw, py - oy * hw, "blue"))
            self.cones.append(Cone(px + ox * hw, py + oy * hw, "yellow"))

        for i in range(4):
            self.cones[i].cone_type = "small_orange"
        self.cones[-2].cone_type = "large_orange"
        self.cones[-1].cone_type = "large_orange"

    def compute_delaunay(self):
        pts = np.array([[c.x, c.y] for c in self.cones])
        simplices = Delaunay(pts).simplices
        self.triangles = [Triangle(tuple(t)) for t in simplices]

    def filter_triangles(self):
        max_edge = self.track_width * 2.5
        self.triangles = [
            t for t in self.triangles
            if t.has_mixed_types(self.cones) and t.max_edge_length(self.cones) <= max_edge
        ]

    def solve_centerline(self) -> List[Tuple[float, float]]:
        cross_edges: Set[Tuple[int, int]] = set()
        for t in self.triangles:
            for i in range(3):
                a, b = t.indices[i], t.indices[(i + 1) % 3]
                ta, tb = self.cones[a].cone_type, self.cones[b].cone_type
                is_blue_yellow = (ta == "blue" and tb == "yellow") or (ta == "yellow" and tb == "blue")
                is_orange_cross = (
                    ta in {"small_orange", "large_orange"} and
                    tb in {"small_orange", "large_orange"} and
                    abs(self.cones[a].y - self.cones[b].y) > 0.01
                )
                if is_blue_yellow or is_orange_cross:
                    cross_edges.add((min(a, b), max(a, b)))

        if not cross_edges:
            return []

        edges_list = list(cross_edges)
        midpts = np.array([
            ((self.cones[e[0]].x + self.cones[e[1]].x) / 2,
             (self.cones[e[0]].y + self.cones[e[1]].y) / 2)
            for e in edges_list
        ])

        adjacency = {i: [] for i in range(len(edges_list))}
        for t in self.triangles:
            edges_in_t = []
            for i in range(3):
                a, b = t.indices[i], t.indices[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                if edge in cross_edges:
                    edges_in_t.append(edges_list.index(edge))
            for i in range(len(edges_in_t)):
                for j in range(i + 1, len(edges_in_t)):
                    idx_i, idx_j = edges_in_t[i], edges_in_t[j]
                    if idx_j not in adjacency[idx_i]:
                        adjacency[idx_i].append(idx_j)
                    if idx_i not in adjacency[idx_j]:
                        adjacency[idx_j].append(idx_i)

        start_pos = np.array([self.cones[0].x, self.cones[0].y])
        start_idx = int(np.argmin(np.linalg.norm(midpts - start_pos, axis=1)))
        sorted_indices = [start_idx]
        remaining = set(range(len(edges_list)))
        remaining.remove(start_idx)

        while remaining:
            curr = sorted_indices[-1]
            candidates = [n for n in adjacency[curr] if n in remaining]
            if candidates:
                dists = np.linalg.norm(midpts[candidates] - midpts[curr], axis=1)
                next_idx = candidates[np.argmin(dists)]
            else:
                rem_list = list(remaining)
                dists = np.linalg.norm(midpts[rem_list] - midpts[curr], axis=1)
                next_idx = rem_list[np.argmin(dists)]
            sorted_indices.append(next_idx)
            remaining.remove(next_idx)

        sorted_pts = midpts[sorted_indices]
        centerline = [(float(p[0]), float(p[1])) for p in sorted_pts]
        if centerline:
            centerline.append(centerline[0])
        self.centerline = centerline
        return centerline

    def smooth_centerline(self, num_points: int = 500) -> List[Tuple[float, float]]:
        if len(self.centerline) < 4:
            self.smoothed_centerline = self.centerline
            return self.centerline

        centerline_np = np.array(self.centerline)
        dx = np.diff(centerline_np[:, 0])
        dy = np.diff(centerline_np[:, 1])
        dist = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(dist)])
        mask = np.hstack([[True], dist > 1e-6])
        s, x, y = s[mask], centerline_np[:, 0][mask], centerline_np[:, 1][mask]

        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            s = np.append(s, s[-1] + np.hypot(x[-2] - x[-1], y[-2] - y[-1]))

        cs_x = CubicSpline(s, x, bc_type='periodic')
        cs_y = CubicSpline(s, y, bc_type='periodic')
        s_new = np.linspace(0, s[-1], num_points)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)
        self.smoothed_centerline = list(zip(x_new, y_new))
        return self.smoothed_centerline

    # -----------------------------------------------------------------------
    # LAP 2+: Minimum Curvature Racing Line
    # -----------------------------------------------------------------------

    def _compute_normals(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute unit outward normals (pointing left / CCW) at each point.
        Normal = 90° CCW rotation of the unit tangent.
        """
        n = len(pts)
        tangents = np.array([pts[(i + 1) % n] - pts[(i - 1) % n] for i in range(n)])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        tangents /= norms
        # Rotate 90° CCW: (tx, ty) → (−ty, tx)
        return np.column_stack([-tangents[:, 1], tangents[:, 0]])

    def _compute_boundary_offsets(
        self, pts: np.ndarray, normals: np.ndarray, safety_margin: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For every centerline point, find the maximum alpha (offset along normal)
        before hitting a left (blue) or right (yellow) cone.

        Positive alpha  → left  (towards blue)
        Negative alpha  → right (towards yellow)

        Returns:
            alpha_max : max offset towards blue side  (≥ 0)
            alpha_min : max offset towards yellow side (≤ 0)
        """
        n = len(pts)
        default_hw = self.track_width / 2.0 - safety_margin

        alpha_max = np.full(n,  default_hw)   # positive = towards blue
        alpha_min = np.full(n, -default_hw)   # negative = towards yellow

        blue_pts = np.array([[c.x, c.y] for c in self.cones if c.cone_type == "blue"])
        yell_pts = np.array([[c.x, c.y] for c in self.cones if c.cone_type == "yellow"])

        if blue_pts.size == 0 or yell_pts.size == 0:
            return alpha_max, alpha_min

        for i in range(n):
            ni = normals[i]   # unit normal (pointing left)

            # --- Left boundary (blue cones, positive side) ---
            proj_blue = (blue_pts - pts[i]) @ ni      # signed projection onto normal
            pos_proj  = proj_blue[proj_blue > 0]
            if pos_proj.size:
                alpha_max[i] = max(np.min(pos_proj) - safety_margin, 0.05)

            # --- Right boundary (yellow cones, negative side) ---
            proj_yell = (yell_pts - pts[i]) @ ni
            neg_proj  = proj_yell[proj_yell < 0]
            if neg_proj.size:
                alpha_min[i] = min(np.max(neg_proj) + safety_margin, -0.05)

        return alpha_max, alpha_min

    def _build_qp_matrices(
        self, pts: np.ndarray, normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the QP cost matrices for minimum curvature.

        The path is parameterised as:
            P_i = C_i + α_i · n_i

        Minimising path curvature ≈ minimising the sum of squared second
        differences:
            cost = Σ_i  ‖ΔΔP_i‖²
                 = Σ_i  ‖ΔΔC_i  +  (B α)_i ‖²

        Written in matrix form:
            cost = ‖d2c + B α‖²
                 = αᵀ (BᵀB) α  +  2 (Bᵀ d2c)ᵀ α  +  const

        where B ∈ R^{2n × n} encodes how α offsets affect second differences,
        and d2c ∈ R^{2n} is the second difference of the centerline.

        Returns:
            Q   = BᵀB           (n × n, positive semi-definite)
            c   = Bᵀ d2c        (n,)
        """
        n   = len(pts)
        nx  = normals[:, 0]
        ny  = normals[:, 1]
        xc  = pts[:, 0]
        yc  = pts[:, 1]

        B   = np.zeros((2 * n, n))
        d2c = np.zeros(2 * n)

        for i in range(n):
            im1, ip1 = (i - 1) % n, (i + 1) % n

            # x-component of second difference at i
            B[2*i,   im1] += nx[im1]
            B[2*i,   i  ] += -2.0 * nx[i]
            B[2*i,   ip1] += nx[ip1]
            d2c[2*i]       = xc[ip1] - 2.0*xc[i] + xc[im1]

            # y-component of second difference at i
            B[2*i+1, im1] += ny[im1]
            B[2*i+1, i  ] += -2.0 * ny[i]
            B[2*i+1, ip1] += ny[ip1]
            d2c[2*i+1]     = yc[ip1] - 2.0*yc[i] + yc[im1]

        Q = B.T @ B          # (n, n)
        c = B.T @ d2c        # (n,)
        return Q, c

    def compute_racing_line(
        self,
        safety_margin: float = 0.5,
        max_iter: int = 2000,
    ) -> List[Tuple[float, float]]:
        """
        Compute the minimum-curvature racing line (Lap 2+).

        Uses the full SLAM cone map (available after lap 1) to solve a
        Quadratic Program:

            min   αᵀ Q α + 2 cᵀ α
            s.t.  alpha_min ≤ α ≤ alpha_max

        where α_i is the lateral offset at waypoint i from the centerline.

        Args:
            safety_margin: Buffer from cone edge to path boundary (m).
            max_iter:       Maximum L-BFGS-B iterations.

        Returns:
            List of (x, y) waypoints for the optimised racing line.
        """
        if len(self.smoothed_centerline) < 4:
            print("[WARNING] Smoothed centerline not available — run smooth_centerline() first.")
            self.racing_line = self.smoothed_centerline
            return self.racing_line

        print("Computing minimum-curvature racing line …")
        pts     = np.array(self.smoothed_centerline)   # (n, 2)
        n       = len(pts)

        # 1. Geometry: unit normals pointing left (CCW)
        normals = self._compute_normals(pts)

        # 2. Per-point lateral bounds from cone positions
        alpha_max, alpha_min = self._compute_boundary_offsets(pts, normals, safety_margin)

        # 3. QP cost matrices
        Q, c_vec = self._build_qp_matrices(pts, normals)

        # 4. Solve bounded QP via L-BFGS-B
        #    cost(α)  = αᵀ Q α + 2 cᵀ α
        #    grad(α)  = 2 Q α  + 2 c
        def cost(alpha: np.ndarray) -> float:
            return float(alpha @ Q @ alpha + 2.0 * c_vec @ alpha)

        def grad(alpha: np.ndarray) -> np.ndarray:
            return 2.0 * (Q @ alpha + c_vec)

        bounds = list(zip(alpha_min, alpha_max))
        alpha0 = np.zeros(n)

        result = minimize(
            cost, alpha0, jac=grad,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            print(f"[WARNING] Optimiser: {result.message}")
        else:
            print(f"[OK] Optimiser converged in {result.nit} iterations.")

        alpha_opt  = result.x
        racing_pts = pts + alpha_opt[:, np.newaxis] * normals
        self.racing_line = [(float(x), float(y)) for x, y in racing_pts]

        # Stats
        mean_offset = np.mean(np.abs(alpha_opt))
        max_offset  = np.max(np.abs(alpha_opt))
        print(f"    Mean lateral offset: {mean_offset:.3f} m  |  Max: {max_offset:.3f} m")
        return self.racing_line

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot(self, plot_triangles=False, plot_centerline=False,
             plot_smooth=True, plot_racing=True):
        """Plot the track, cones, smoothed centerline, and racing line."""
        color_map = {
            "blue": "blue", "yellow": "gold",
            "small_orange": "orange", "large_orange": "darkorange"
        }
        size_map = {"blue": 25, "yellow": 25, "small_orange": 80, "large_orange": 150}

        fig, ax = plt.subplots(figsize=(14, 9))
        pts = np.array([[c.x, c.y] for c in self.cones])

        if plot_triangles:
            for t in self.triangles:
                tp = pts[list(t.indices)]
                ax.fill(tp[:, 0], tp[:, 1], color="lightgrey", alpha=0.25, zorder=1)
                ax.plot(np.append(tp[:, 0], tp[0, 0]), np.append(tp[:, 1], tp[0, 1]),
                        color="grey", lw=0.5, zorder=2)

        for c in self.cones:
            ax.scatter(c.x, c.y,
                       color=color_map.get(c.cone_type, "black"),
                       s=size_map.get(c.cone_type, 25),
                       edgecolors="black", linewidths=0.3, zorder=5)

        if plot_centerline and self.centerline:
            cx, cy = zip(*self.centerline)
            ax.plot(cx, cy, color="red", lw=1.5, zorder=6,
                    label=f"Raw Centerline ({len(self.centerline)} pts)")

        if plot_smooth and self.smoothed_centerline:
            sx, sy = zip(*self.smoothed_centerline)
            ax.plot(sx, sy, color="limegreen", lw=2.0, zorder=7,
                    label=f"Lap 1 — Smoothed Centerline ({len(self.smoothed_centerline)} pts)")

        if plot_racing and self.racing_line:
            rx, ry = zip(*self.racing_line)
            ax.plot(rx, ry, color="orangered", lw=2.5, zorder=8,
                    label=f"Lap 2+ — Min-Curvature Racing Line ({len(self.racing_line)} pts)")

        ax.set_aspect("equal")
        ax.set_title("FS Autocross — Centerline (Lap 1) vs Racing Line (Lap 2+)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("track_output.png", dpi=150, bbox_inches="tight")
        print("Plot saved → track_output.png")
        plt.show()

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    def run(self):
        """Full pipeline: build → triangulate → filter → centerline → smooth → racing line → plot."""
        self.build_cones()
        self.compute_delaunay()
        self.filter_triangles()
        self.solve_centerline()
        # 200 pts: fine enough for smooth output, small enough for fast QP convergence
        self.smooth_centerline(num_points=200)

        # --- Lap 2+: compute racing line from full cone map ---
        self.compute_racing_line(safety_margin=0.5)

        self.plot(plot_triangles=False, plot_centerline=False,
                  plot_smooth=True, plot_racing=True)

        print(f"\nSummary:")
        print(f"  Cones      : {len(self.cones)}")
        print(f"  Triangles  : {len(self.triangles)}")
        print(f"  Centerline : {len(self.centerline)} pts (raw)  |  "
              f"{len(self.smoothed_centerline)} pts (smoothed)")
        print(f"  Racing line: {len(self.racing_line)} pts")


if __name__ == "__main__":
    Track().run()
