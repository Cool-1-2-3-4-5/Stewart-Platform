import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# Global Variables
CAMERA_INDEX = 0                # Pi 5 camera port: 0 = CAM/DISP 0, 1 = CAM/DISP 1
FRAME_W, FRAME_H = 640, 480     # Low resolution keeps capture + detection fast on a Pi. Sensor is 16:9, so 4:3 crops the sides
TARGET_FPS = 60                 # Fixed frame rate; also caps exposure time at 1/60 s
CAMERA_SETTLE_S = 1.0           # Time for auto-exposure / white balance to settle before locking
LOCK_EXPOSURE = True            # Freeze exposure + white balance after startup so HSV colours stay constant
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

MIN_QUAD_AREA_PX = 2000.0       # Reject small tag layouts

# Red ball (HSV, OpenCV H 0-179). PLACEHOLDERS: tune under the real lighting
RED_LOW_1 = np.array([0, 100, 100], dtype=np.uint8)
RED_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_LOW_2 = np.array([170, 100, 100], dtype=np.uint8)
RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)
# Ignore small noise
MIN_BALL_AREA_PX = 30.0

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Filters out small glares or edge pixels on the ball, small specks filtered due to the minimum area pre-filter

def open_camera():
    cam = Picamera2(CAMERA_INDEX)
    frame_us = int(1_000_000 / TARGET_FPS)
    config = cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},  # "RGB888" is stored B,G,R: what OpenCV expects
        controls={"FrameDurationLimits": (frame_us, frame_us)},
        queue=False,  # capture_array() waits for a fresh frame instead of returning a queued, older one
    )
    cam.configure(config)
    cam.start()
    time.sleep(CAMERA_SETTLE_S)  # let auto-exposure and white balance settle

    # Focus once on the maze, then hold the lens there so it doesn't refocus while the platform tilts
    cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
    if not cam.autofocus_cycle():
        print("[camera] Autofocus did not converge; check the image is sharp")

    if LOCK_EXPOSURE:
        md = cam.capture_metadata()
        cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": md["ExposureTime"],
            "AnalogueGain": md["AnalogueGain"],
            "ColourGains": md["ColourGains"],
        })
        print(f"[camera] Locked exposure {md['ExposureTime']} us, gain {md['AnalogueGain']:.2f}")

    print(f"[camera] Pi Camera {FRAME_W}x{FRAME_H} @ {TARGET_FPS} FPS")
    return cam

def create_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement to ensure consistent results
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


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
    if not cv2.isContourConvex(src_points.reshape(-1, 1, 2)): # ensures lines connected them are not twisted (no bowties)
        return None
    signed_area = cv2.contourArea(src_points, oriented=True)
    if abs(signed_area) < MIN_QUAD_AREA_PX:
        return None
    dst_area = cv2.contourArea(DST_POINTS, oriented=True) # ensures shape isnt mirroed to prevent maze from going left-to-right
    if np.sign(signed_area) != np.sign(dst_area):
        return None
    return cv2.getPerspectiveTransform(src_points, DST_POINTS) # better than findhomography rfunction beacuse the others are really useful when more than 4 points

def detect_ball(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask_low = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    mask_high = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.bitwise_or(mask_low, mask_high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)  # Kill speckle noise
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: # No ball found
        return None

    # Assume the largest matching blob is the ball.
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BALL_AREA_PX:
        return None

    # centroid to account for specs and imperfections
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"] # centroidial in x
    cy = m["m01"] / m["m00"] # centroidial posityion in y
    return (cx, cy)