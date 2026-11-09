import cv2
from Computer_vision import open_camera, create_aruco_detector, get_tag_centers, compute_homography, WARP_SIZE

TAG_SPAN_MM = 170.0 # centres of Aruco tags
DILATE_MM = 5.5
WALL_THRESHOLD = 127


def Maze_Reader():
    cam = open_camera()
    frame = cam.capture_array()
    cam.close()  # free the camera
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
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))  # blue range, tune if needed
    binary[blue > 0] = 0
    cv2.imwrite("maze_binary.png", binary * 255)

    px_per_mm = WARP_SIZE / TAG_SPAN_MM
    r = max(1, round(DILATE_MM * px_per_mm))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(binary, kernel)
    cv2.imwrite("maze_dilated.png", dilated * 255)

    contours, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h, x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[0]), cv2.boundingRect(sorted_contours[1])
    mid_point_1 = (x+(w/2),y+(h/2))
    mid_point_2 = (x2+(w2/2),y2+(h2/2))
    if mid_point_1[0] >= mid_point_1[1]:
        temp = mid_point_1
        mid_point_1 = mid_point_2
        mid_point_2 = temp
    return dilated.tolist(), mid_point_1, mid_point_2
