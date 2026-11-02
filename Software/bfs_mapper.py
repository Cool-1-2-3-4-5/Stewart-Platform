from collections import deque
import numpy as np

def bfs_path_dir(bfs_list):
    start = (0,0)
    double_queue = deque([start])
    visited = {start}
    path = {}
    path[start] = None
    target_cell = (26,26)
    while double_queue:
        cur_cell = double_queue.popleft()
        if cur_cell == target_cell or len(visited) >= 27*27:
            break
        x, y = cur_cell
        if checker((x, y - 1),visited,bfs_list): # Up
            double_queue.append((x, y - 1))
            path[(x, y - 1)] = cur_cell
            visited.add((x, y - 1))

        if checker((x + 1, y),visited,bfs_list): # Right
            double_queue.append((x + 1, y))
            path[(x + 1, y)] = cur_cell
            visited.add((x + 1, y))

        if checker((x, y + 1),visited,bfs_list): # Down
            double_queue.append((x, y + 1))
            path[(x, y + 1)] = cur_cell
            visited.add((x, y + 1))

        if checker((x - 1, y),visited,bfs_list): # Left
            double_queue.append((x - 1, y))
            path[(x - 1, y)] = cur_cell
            visited.add((x - 1, y))

    if target_cell not in path:
        return None, None

    first_pass = []
    path_finder = []
    runner = target_cell
    while runner!=start:
        first_pass.append(runner)
        runner = path[runner]
    first_pass.append(start)
    while len(first_pass)!=0:
        path_finder.append(first_pass.pop())

    directions = []
    for i in range(len(path_finder)-1):
        x, y = path_finder[i] # current cell
        nx, ny = path_finder[i+1] # next cell
        step = (nx - x, ny - y)
        if step == (0, -1):
            directions.append("up")
        elif step == (1, 0):
            directions.append("right")
        elif step == (0, 1):
            directions.append("down")
        elif step == (-1, 0):
            directions.append("left")
    
    return path_finder, directions

def checker(coordinates_,visited_set,bfs_list):
    if coordinates_[0] > 26 or coordinates_[0] < 0 or coordinates_[1] > 26 or coordinates_[1] < 0:
        return False
    elif bfs_list[coordinates_[1]][coordinates_[0]] == 1:
        return False
    elif coordinates_ in visited_set:
        return False
    else:
        return True