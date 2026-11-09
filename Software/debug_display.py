import cv2

from Computer_vision import SHOW_DEBUG


def show(warped, waypoints, target, ball_xy):
    """Draw the path, current target and ball. Returns False if 'q' was pressed."""
    if not SHOW_DEBUG or warped is None:
        return True
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        cv2.line(warped, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.circle(warped, (int(target[0]), int(target[1])), 4, (255, 0, 255), -1)
    if ball_xy is not None:
        cv2.circle(warped, (int(ball_xy[0]), int(ball_xy[1])), 6, (0, 255, 0), 2)
    cv2.imshow("main", warped)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def close():
    cv2.destroyAllWindows()
