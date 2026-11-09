from arduino_link import open_link, send
from Maze_Analyser import Maze_Reader
from bfs_mapper import bfs_path_dir
from Computer_vision import open_camera, create_aruco_detector
from controller import PIDmove
from ik_core import inverse_kinematics

KP, KI, KD = 0.05, 0.0, 0.5   # Tune on the build

def main():
    link = open_link()
    level = inverse_kinematics(0, 0, 0, 0, 0, 0)
    send(link, level)

    grid, start_xy, goal_xy = Maze_Reader()
    waypoints, _ = bfs_path_dir(grid, start_xy, goal_xy)
    if waypoints is None:
        raise RuntimeError(f"No path from {START_XY} to {GOAL_XY}; check maze_dilated.png")

    cam = open_camera()
    detector = create_aruco_detector()
    try:
        for target in waypoints[1:]:
            PIDmove(cam, detector, link, target, KP, KI, KD)
        print("Goal reached!")
    finally:
        send(link, level)
        link.close()
        cam.close()

if __name__ == "__main__":
    main()