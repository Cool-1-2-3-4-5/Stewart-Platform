import cv2
from Computer_vision import create_aruco_detector, get_tag_centers, compute_homography, WARP_SIZE

cap = cv2.VideoCapture(0)

def Maze_Reader():
    ok, frame = cap.read()
    cap.release()
    cv2.imwrite("maze_snapshot.png", frame)
    detector = create_aruco_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    centres = get_tag_centers(corners, ids)
    H = compute_homography(centres)
    if H is None:
        raise RuntimeError("All 4 tags are not visible")
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    _, binary = cv2.threshold(warped, 127, 1, cv2.THRESH_BINARY_INV)
    cv2.imwrite("maze_binary.png", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dilated = cv2.dilate(binary, kernel)
    cv2.imwrite("maze_dilated.png", dilated)
    pure_maze = warped_frame[15:285, 15:285] # remove aruco tags from (300,300)
    # Downsampling
    small_grid_img = cv2.resize(pure_maze, (27, 27), interpolation=cv2.INTER_AREA)
    occupancy_grid = np.where(small_grid_img > 12, 1, 0) # convert to array to run BFS
    bfs_list = occupancy_grid.tolist()
    return bfs_list