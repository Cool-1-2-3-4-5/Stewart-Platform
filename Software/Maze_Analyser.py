import cv2
from Computer_vision import open_camera, create_aruco_detector, get_tag_centers, compute_homography, WARP_SIZE

TAG_SPAN_MM = 170.0 # centres of Aruco tags
DILATE_MM = 5.5
WALL_THRESHOLD = 127


def Maze_Reader():
    cam = open_camera()
    frame = cam.capture_array()
    cam.close()  # free the camera so Computer_vision.main() can use it
    cv2.imwrite("maze_snapshot.png", frame)

    detector = create_aruco_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    centres = get_tag_centers(corners, ids)
    H = compute_homography(centres)
    if H is None:
        raise RuntimeError("All 4 tags are not visible")
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(warped_gray, WALL_THRESHOLD, 1, cv2.THRESH_BINARY_INV)
    cv2.imwrite("maze_binary.png", binary * 255)  # * 255 so walls show white in the saved image

    px_per_mm = WARP_SIZE / TAG_SPAN_MM
    r = max(1, round(DILATE_MM * px_per_mm)) # how much to dilate in pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(binary, kernel)
    cv2.imwrite("maze_dilated.png", dilated * 255)
    return dilated.tolist()
