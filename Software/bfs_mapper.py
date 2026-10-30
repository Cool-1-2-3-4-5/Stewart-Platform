"""
bfs_mapper.py - One-shot BFS shortest path through a warped maze image.

Runs once at startup. Input is the 300x300 top-down view produced by the
vision pipeline (white walls on a black floor). Output is an ordered (N, 2)
array of (x, y) pixel waypoints from start to goal through the black floor.

Pipeline:
    1. Threshold: bright pixels = wall.
    2. Grow walls by the ball radius so the path keeps the ball off the walls.
    3. Discretize into CELL_PX x CELL_PX cells; a cell is free if it is
       (almost) wall-free.
    4. 4-connected BFS from the start cell to the goal cell (deque queue).
       Every move costs the same, so the first time BFS reaches the goal it
       has found a shortest path in cell steps.
    5. Walk the parent links back from goal to start and convert cells to
       pixel centres.
"""

from collections import deque

import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
CELL_PX = 10                  # Grid cell size [px]; 300 / 10 -> 30x30 grid
WALL_THRESHOLD = 128          # Grayscale > this counts as wall (white)
WALL_CLEARANCE_PX = 6         # Grow walls by ~ball radius [px]; 0 to disable
CELL_BLOCKED_FRACTION = 0.2   # Cell is blocked if > this fraction is wall

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right


# =============================================================================
# Grid construction
# =============================================================================

def build_free_grid(warped):
    """Return a (rows, cols) bool grid: True = free floor cell."""
    gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    walls = (gray > WALL_THRESHOLD).astype(np.uint8)

    if WALL_CLEARANCE_PX > 0:
        k = 2 * WALL_CLEARANCE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        walls = cv2.dilate(walls, kernel)

    rows, cols = walls.shape[0] // CELL_PX, walls.shape[1] // CELL_PX
    walls = walls[:rows * CELL_PX, :cols * CELL_PX]
    # Split into (rows, CELL_PX, cols, CELL_PX) blocks and average each block.
    wall_fraction = walls.reshape(rows, CELL_PX, cols, CELL_PX).mean(axis=(1, 3))
    return wall_fraction <= CELL_BLOCKED_FRACTION


def pixel_to_cell(xy, grid_shape):
    """(x, y) pixel -> (row, col) cell, clamped to the grid."""
    rows, cols = grid_shape
    col = int(np.clip(xy[0] // CELL_PX, 0, cols - 1))
    row = int(np.clip(xy[1] // CELL_PX, 0, rows - 1))
    return row, col


def cell_to_pixel(cell):
    """(row, col) cell -> (x, y) pixel at the cell centre."""
    row, col = cell
    return (col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX


def nearest_free_cell(free, cell):
    """Snap a cell to the closest free cell (start/goal may sit in a wall margin)."""
    if free[cell]:
        return cell
    candidates = np.argwhere(free)
    if len(candidates) == 0:
        raise RuntimeError("Grid has no free cells; check WALL_THRESHOLD / lighting.")
    d2 = np.sum((candidates - np.array(cell)) ** 2, axis=1)
    r, c = candidates[np.argmin(d2)]
    return int(r), int(c)


# =============================================================================
# Search
# =============================================================================

def bfs(free, start, goal):
    """
    Shortest 4-connected path on the free grid, as a list of (row, col)
    from start to goal, or None if the goal is unreachable.
    """
    rows, cols = free.shape
    visited = np.zeros((rows, cols), dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    if not visited[goal]:
        return None

    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def simplify_path(cells):
    """Keep only start, goal and turning points (drops collinear cells)."""
    if len(cells) < 3:
        return list(cells)
    out = [cells[0]]
    for prev, cur, nxt in zip(cells, cells[1:], cells[2:]):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            out.append(cur)
    out.append(cells[-1])
    return out


def map_maze(warped, start_xy, goal_xy, simplify=False):
    """
    Compute the shortest path once at startup.

    warped    300x300 BGR or grayscale top-down maze image
    start_xy  (x, y) pixel start, e.g. the ball position from the tracker
    goal_xy   (x, y) pixel target
    simplify  True -> return only corner waypoints instead of every cell

    Returns an (N, 2) float32 array of (x, y) pixel waypoints, start -> goal.
    Raises RuntimeError if no path exists.
    """
    free = build_free_grid(warped)
    start = nearest_free_cell(free, pixel_to_cell(start_xy, free.shape))
    goal = nearest_free_cell(free, pixel_to_cell(goal_xy, free.shape))

    cells = bfs(free, start, goal)
    if cells is None:
        raise RuntimeError("No path from start to goal; check walls/clearance.")
    if simplify:
        cells = simplify_path(cells)
    return np.array([cell_to_pixel(c) for c in cells], dtype=np.float32)


# =============================================================================
# Visualization
# =============================================================================

def draw_path(image, waypoints, start_xy=None, goal_xy=None):
    """Return a copy of the image with the waypoint path drawn on it."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    pts = np.round(waypoints).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    if start_xy is not None:
        cv2.circle(vis, (int(start_xy[0]), int(start_xy[1])), 7, (0, 255, 0), 2)
    if goal_xy is not None:
        cv2.circle(vis, (int(goal_xy[0]), int(goal_xy[1])), 7, (255, 0, 255), 2)
    return vis


if __name__ == "__main__":
    # Synthetic test maze: border walls plus two baffles.
    maze = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(maze, (0, 0), (299, 299), (255, 255, 255), 8)
    cv2.line(maze, (100, 0), (100, 200), (255, 255, 255), 8)
    cv2.line(maze, (200, 100), (200, 299), (255, 255, 255), 8)

    start, goal = (40, 260), (260, 40)
    path = map_maze(maze, start, goal, simplify=True)
    print(f"{len(path)} waypoints:\n{path}")
    cv2.imshow("BFS path", draw_path(maze, path, start, goal))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
