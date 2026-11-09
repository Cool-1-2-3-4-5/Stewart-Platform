import cv2

from Computer_vision import get_tag_centers, compute_homography, detect_ball, WARP_SIZE


def find_ball(cam, detector):
    frame = cam.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    H = compute_homography(get_tag_centers(corners, ids))
    if H is None:
        return None
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    return detect_ball(warped)
