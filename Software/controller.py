import time
from arduino_link import send
from ik_core import inverse_kinematics
from ball_tracker import find_ball
MAX_TIME = 5            # seconds allowed to reach one waypoint
POSITION_TOLERANCE = 12 # px, waypoint reached when the ball is this close
MAX_TILT = 6.0          # deg, hard limit on roll and pitch

# Tilt direction fixes: flip a sign if the ball rolls the wrong way on that axis
SIGN_X = 1
SIGN_Y = 1


def PIDmove(cam, detector, link, target, kp, ki, kd):
    cumerror_x = 0
    cumerror_y = 0
    prev_error_x = 0
    prev_error_y = 0
    start = time.monotonic()

    while time.monotonic() - start < MAX_TIME:
        ball_xy = find_ball(cam, detector)
        if ball_xy is None:  # ball lost: level and keep looking
            send(link, inverse_kinematics(0, 0, 0, 0, 0, 0))
            continue

        error_x = target[0] - ball_xy[0]
        error_y = target[1] - ball_xy[1]

        distance = (error_x ** 2 + error_y ** 2) ** 0.5
        if distance <= POSITION_TOLERANCE:
            break

        cumerror_x += error_x
        cumerror_y += error_y
        derivate_difference_x = error_x - prev_error_x
        derivate_difference_y = error_y - prev_error_y

        tilt_x = kp * error_x + ki * cumerror_x + kd * derivate_difference_x
        tilt_y = kp * error_y + ki * cumerror_y + kd * derivate_difference_y
        tilt_x = max(-MAX_TILT, min(MAX_TILT, tilt_x))
        tilt_y = max(-MAX_TILT, min(MAX_TILT, tilt_y))

        cmds = inverse_kinematics(0, 0, 0, SIGN_Y * tilt_y, SIGN_X * tilt_x, 0)  # roll, pitch
        if cmds is not None:
            send(link, cmds)

        prev_error_x = error_x
        prev_error_y = error_y

    send(link, inverse_kinematics(0, 0, 0, 0, 0, 0))  # stop: level the platform
