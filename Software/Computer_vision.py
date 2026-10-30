"""
Stewart Platform Maze Solver - Vision Pipeline
==============================================

Pipeline (per frame):
    1. Grab a 640x480 frame from the webcam (low latency, high FPS).
    2. Detect ArUco tags (DICT_4X4_50) sitting at the midpoints of the maze's 4 sides.
    3. Compute each tag's center by averaging its 4 corners.
    4. Map tag IDs to fixed physical sides (1=Top, 2=Right, 3=Bottom, 4=Left).
    5. Build a homography from the 4 midpoints (a diamond) to a flat 300x300 grid.
    6. Warp the frame into a top-down view of the maze.
    7. HSV-threshold the warped view to isolate the ball and find its (x, y) center.

The ball position (in warped 300x300 maze coordinates) is exposed as `ball_xy`
inside the main loop, ready to feed into BFS waypoint tracking / RL inference.

Press 'q' in any debug window to quit.
"""

import time
import cv2
import numpy as np

# Global Variables
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480     # Low resolution keeps capture + detection fast on a Pi
TARGET_FPS = 60                 # Requested; the camera will clamp to what it supports
WARP_SIZE = 300                 # Output top-down view is WARP_SIZE x WARP_SIZE pixels

# Using differnt ideas since Yaw would change their positions
TAG_ID_TOP = 1
TAG_ID_RIGHT = 2
TAG_ID_BOTTOM = 3
TAG_ID_LEFT = 4

TAG_ORDER = (TAG_ID_TOP, TAG_ID_RIGHT, TAG_ID_BOTTOM, TAG_ID_LEFT)


# Ideal positions of the warped frame
_HALF = WARP_SIZE / 2.0
DST_POINTS = np.array(
    [
        [_HALF, 0.0],               # Top    (ID 1)
        [float(WARP_SIZE), _HALF],  # Right  (ID 2)
        [_HALF, float(WARP_SIZE)],  # Bottom (ID 3)
        [0.0, _HALF],               # Left   (ID 4)
    ],
    dtype=np.float32,
)

H_MAX_AGE_S = 0.5               # Reuse latest good matrix for at most this long
MIN_QUAD_AREA_PX = 2000.0       # Reject small tag layouts

BALL_COLOR = "orange"
HSV_BOUNDS = {
    "orange": (np.array([5, 120, 120], dtype=np.uint8),
               np.array([25, 255, 255], dtype=np.uint8)),
    "white":  (np.array([0, 0, 200], dtype=np.uint8),
               np.array([179, 50, 255], dtype=np.uint8)),
}
# Ignore small noise
MIN_BALL_AREA_PX = 30.0

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Filters out small glares or edge pixels on the ball, small specks filtered due to the minimum area pre-filter

# Display Toggle
SHOW_DEBUG = True



# Startup Functions

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # MJPG lets most USB webcams deliver higher FPS than raw YUYV at 640x480.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Keep the driver queue short so we always process the newest frame, not a
    # stale one. Not every backend honors this, but it's harmless if ignored.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[camera] {actual_w}x{actual_h} @ {actual_fps:.0f} FPS (requested)")
    return cap


def create_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement to ensure consistent results
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)



# Frame Processing Functions

def get_tag_centers(corners, ids):
    centers = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            tag_id = int(marker_id)
            if tag_id in TAG_ORDER:
                centers[tag_id] = marker_corners.reshape(4, 2).mean(axis=0)
    return centers


def compute_homography(centers):
    for tag_id in TAG_ORDER:
        if tag_id not in centers:
            return None
    src_points = np.array([centers[1],centers[2],centers[3],centers[4]], dtype=np.float32)
    if not cv2.isContourConvex(src.reshape(-1, 1, 2)): # ensures lines connected them are not twisted (no bowties)
        return None
    signed_area = cv2.contourArea(src_points, oriented=True)
    if abs(signed_area) < MIN_QUAD_AREA_PX:
        return None
    dst_area = cv2.contourArea(DST_POINTS, oriented=True) # ensures shape isnt mirroed to prevent maze from going left-to-right
    if np.sign(signed_area) != np.sign(dst_area):
        return None

    return cv2.getPerspectiveTransform(src_points, DST_POINTS) # better than findhomography rfunction beacuse the others are really useful when more than 4 points


def detect_ball(warped, lower, upper):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)  # Kill speckle noise
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: # No ball found
        return None, mask

    # Assume the largest matching blob is the ball.
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BALL_AREA_PX:
        return None, mask

    # centroid to account for specs and imperfections
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None, mask
    cx = m["m10"] / m["m00"] # centroidial in x
    cy = m["m01"] / m["m00"] # centroidial posityion in y
    _, radius = cv2.minEnclosingCircle(largest) # ball radius
    return ((cx, cy), radius), mask

# Main

def main():
    cap = open_camera()
    aruco_detector = create_aruco_detector()
    lower, upper = HSV_BOUNDS[BALL_COLOR]

    # Homography fallback state
    last_H = None
    last_H_time = 0.0

    # FPS measurement
    fps = 0.0
    prev_t = time.monotonic()

    consecutive_read_failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures > 30:
                    print("[camera] Lost camera feed, exiting.")
                    break
                continue
            consecutive_read_failures = 0
            now = time.monotonic()

            # ArUco detection (grayscale is needed)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            # Tag centers keyed by physical side
            centers = get_tag_centers(corners, ids)

            # Homography with fallback
            H = compute_homography(centers)
            if H is not None:
                last_H, last_H_time = H, now
                h_status = "LIVE"
            elif last_H is not None and (now - last_H_time) <= H_MAX_AGE_S:
                H = last_H
                h_status = "HELD"
            else:
                H = None
                h_status = "LOST"

            # Warp + ball tracking
            ball_xy = None   # (x, y) in 300x300 maze coords -> feed to BFS/RL
            warped = None
            ball_mask = None
            if H is not None:
                warped = cv2.warpPerspective(
                    frame, H, (WARP_SIZE, WARP_SIZE), flags=cv2.INTER_LINEAR
                )
                ball, ball_mask = detect_ball(warped, lower, upper)
                if ball is not None:
                    (bx, by), radius = ball
                    ball_xy = (bx, by)
                    cv2.circle(warped, (int(bx), int(by)), max(int(radius), 3), (0, 255, 0), 2)
                    cv2.circle(warped, (int(bx), int(by)), 2, (0, 0, 255), -1)

            # TODO: hand ball_xy (and h_status) to the waypoint tracker / RL
            # policy here, then stream servo commands to the Arduino over serial.

            # FPS calculation
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # Debug display
            if SHOW_DEBUG:
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for tag_id, (cx, cy) in centers.items():
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 255), -1)

                status_color = {"LIVE": (0, 255, 0), "HELD": (0, 200, 255), "LOST": (0, 0, 255)}
                cv2.putText(frame, f"H: {h_status}  tags: {sorted(centers)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color[h_status], 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Camera", frame)

                if warped is not None:
                    label = (f"ball: ({ball_xy[0]:.0f}, {ball_xy[1]:.0f})"
                             if ball_xy else "ball: --")
                    cv2.putText(warped, label, (5, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imshow("Top-Down Maze", warped)
                    cv2.imshow("Ball Mask", ball_mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
