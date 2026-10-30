"""
ik_core.py - Inverse kinematics for a 6-servo rotary Stewart platform.

Converts a desired platform pose (x, y, z, roll, pitch, yaw) into six servo
commands in degrees, ready to send to the ESP32.

Frames and units
----------------
Base frame {B}:     origin at the centre of the servo-shaft circle, +Z up.
Platform frame {P}: origin at the centre of the top-plate joint circle, +Z up.
At the home pose, {P} sits h0 above {B} with no rotation and every horn is
horizontal. Pose inputs are offsets from that home pose.
Lengths are millimetres. Public angles are degrees; internal math is radians.
"""

import numpy as np

# =============================================================================
# Physical configuration -- PLACEHOLDERS: measure these on the WATiMake build
# =============================================================================
BASE_RADIUS_MM = 100.0       # Base centre -> servo shaft axis (horn pivot)
PLATFORM_RADIUS_MM = 80.0    # Top-plate centre -> rod-end ball joint centre
HORN_LENGTH_MM = 25.0        # Servo shaft axis -> horn ball joint centre
ROD_LENGTH_MM = 150.0        # Ball joint centre -> ball joint centre

# Anchors come in 3 pairs. Each pair is centred at an angle (CCW from base +X,
# viewed from above); its two anchors sit at centre - half_sep, centre + half_sep.
# Servo index order: pair0(-,+), pair1(-,+), pair2(-,+)  ->  servos 0..5.
BASE_PAIR_CENTERS_DEG = (90.0, 210.0, 330.0)
PLATFORM_PAIR_CENTERS_DEG = (90.0, 210.0, 330.0)
BASE_HALF_SEP_DEG = 20.0
PLATFORM_HALF_SEP_DEG = 8.0

# Each horn swings in a vertical plane tangent to the base circle.
# True  -> the two horns of a pair point away from each other.
# False -> they point toward each other.
HORNS_POINT_OUTWARD = True

# Servo command mapping:  cmd_deg = NEUTRAL + DIRECTION * horn_angle_deg
#   NEUTRAL   = command at which that horn is exactly horizontal
#   DIRECTION = +1 if increasing the command raises the horn tip, else -1
#               (the mirrored servos of a pair normally have opposite signs)
SERVO_NEUTRAL_DEG = (90.0, 90.0, 90.0, 90.0, 90.0, 90.0)
SERVO_DIRECTION = (1, -1, 1, -1, 1, -1)

# Hard servo limits (np.clip is applied to every command).
SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

# Pose envelope, max |offset| from home: x, y, z [mm], roll, pitch, yaw [deg].
POSE_LIMITS = (20.0, 20.0, 15.0, 15.0, 15.0, 20.0)


# =============================================================================
# Math helpers
# =============================================================================

def paired_angles(centers_deg, half_sep_deg):
    """Return 6 anchor angles [rad]: each pair centre c yields (c - d, c + d)."""
    centers = np.repeat(np.radians(centers_deg), 2)
    offsets = np.tile([-1.0, 1.0], 3) * np.radians(half_sep_deg)
    return centers + offsets


def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    Rotation matrix R that takes platform-frame vectors into the base frame.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), i.e. roll about X first,
    then pitch about Y, then yaw about Z, all about the fixed base axes.

        Rx(phi)   = [[1,  0,     0   ],     Ry(theta) = [[ ct, 0, st],
                     [0,  cphi, -sphi],                  [ 0,  1, 0 ],
                     [0,  sphi,  cphi]]                  [-st, 0, ct]]

        Rz(psi)   = [[cpsi, -spsi, 0],
                     [spsi,  cpsi, 0],
                     [0,     0,    1]]

    Multiplied out (c = cos, s = sin; phi = roll, theta = pitch, psi = yaw):

        R = [[c_psi c_th,  c_psi s_th s_phi - s_psi c_phi,  c_psi s_th c_phi + s_psi s_phi],
             [s_psi c_th,  s_psi s_th s_phi + c_psi c_phi,  s_psi s_th c_phi - c_psi s_phi],
             [-s_th,       c_th s_phi,                      c_th c_phi                    ]]

    R is orthonormal (R^-1 = R^T, det R = +1), so it rotates without
    stretching: every platform joint keeps its distance from the centre.
    """
    phi, theta, psi = np.radians([roll_deg, pitch_deg, yaw_deg])
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
    rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


# =============================================================================
# IK solver
# =============================================================================

class StewartIK:
    """
    Rotary-servo Stewart platform inverse kinematics.

    Geometry per leg i (all vectors in the base frame):
        b_i   servo shaft pivot          = R_b (cos theta_bi, sin theta_bi, 0)
        p_i   top joint, platform frame  = R_p (cos theta_pi, sin theta_pi, 0)
        beta_i  azimuth of the vertical plane the horn swings in
        a     horn length,  s  rod length
    """

    def __init__(
        self,
        base_radius=BASE_RADIUS_MM,
        platform_radius=PLATFORM_RADIUS_MM,
        horn_length=HORN_LENGTH_MM,
        rod_length=ROD_LENGTH_MM,
        base_angles_rad=None,
        platform_angles_rad=None,
        horn_plane_rad=None,
        servo_neutral_deg=SERVO_NEUTRAL_DEG,
        servo_direction=SERVO_DIRECTION,
        pose_limits=POSE_LIMITS,
    ):
        self.a = float(horn_length)
        self.s = float(rod_length)

        theta_b = (paired_angles(BASE_PAIR_CENTERS_DEG, BASE_HALF_SEP_DEG)
                   if base_angles_rad is None else np.asarray(base_angles_rad, float))
        theta_p = (paired_angles(PLATFORM_PAIR_CENTERS_DEG, PLATFORM_HALF_SEP_DEG)
                   if platform_angles_rad is None else np.asarray(platform_angles_rad, float))

        if horn_plane_rad is None:
            # The tangent to the base circle at angle theta is theta +/- 90 deg.
            # Servo 2k sits on the clockwise side of its pair, so "outward" for
            # it is the clockwise tangent (theta - 90 deg), and vice versa.
            side = np.tile([-1.0, 1.0], 3)
            if not HORNS_POINT_OUTWARD:
                side = -side
            horn_plane_rad = theta_b + side * (np.pi / 2.0)
        self.beta = np.asarray(horn_plane_rad, float)
        self.cos_beta = np.cos(self.beta)
        self.sin_beta = np.sin(self.beta)

        zeros = np.zeros(6)
        self.b = np.column_stack([base_radius * np.cos(theta_b),
                                  base_radius * np.sin(theta_b), zeros])
        self.p = np.column_stack([platform_radius * np.cos(theta_p),
                                  platform_radius * np.sin(theta_p), zeros])

        self.servo_neutral = np.asarray(servo_neutral_deg, float)
        self.servo_direction = np.asarray(servo_direction, float)
        self.pose_limits = np.asarray(pose_limits, float)
        self.h0 = self._home_height()

    def _home_height(self):
        """
        Height h0 of {P} above {B} at which every horn is horizontal.

        With the horn horizontal (alpha = 0) its tip is
            H_i = b_i + a (cos beta_i, sin beta_i, 0).
        The rod joins H_i to the top joint (p_xi, p_yi, h0) and has length s:
            (p_xi - H_xi)^2 + (p_yi - H_yi)^2 + h0^2 = s^2
            h0 = sqrt(s^2 - d_i^2),  d_i = horizontal distance tip -> joint.
        A symmetric build gives the same h0 for all six legs. For a measured,
        slightly asymmetric build the mean is used, so home horns are only
        approximately horizontal.
        """
        tips = self.b[:, :2] + self.a * np.column_stack([self.cos_beta, self.sin_beta])
        d2 = np.sum((self.p[:, :2] - tips) ** 2, axis=1)
        if np.any(d2 >= self.s ** 2):
            raise ValueError("Rod too short for this geometry: no valid home height.")
        return float(np.mean(np.sqrt(self.s ** 2 - d2)))

    def leg_vectors(self, pose):
        """
        Vector l_i from each servo pivot to its top joint, for a pose.

        Translation + rotation (rigid-body transform of each top joint):
            q_i = T + R p_i,   T = (x, y, h0 + z)
            l_i = q_i - b_i
        R p_i rotates the joint about the platform centre; T then moves the
        rotated platform to its commanded position above the base.
        """
        x, y, z, roll, pitch, yaw = pose
        rot = rotation_matrix(roll, pitch, yaw)
        t = np.array([x, y, self.h0 + z])
        q = t + self.p @ rot.T          # row-wise R @ p_i for all 6 joints
        return q - self.b

    def horn_angles(self, pose):
        """
        Horn angle alpha_i [rad] above horizontal for each servo, or None if
        any leg cannot reach (pose outside the workspace).

        Derivation: the horn tip swings on a circle of radius a,
            H_i = b_i + a (cos(alpha) cos(beta), cos(alpha) sin(beta), sin(alpha)).
        The rod is rigid, so |q_i - H_i| = s. Expanding with l = q_i - b_i:
            |l|^2 - 2a [sin(alpha) l_z + cos(alpha)(cos(beta) l_x + sin(beta) l_y)] + a^2 = s^2
        which is linear in sin and cos of alpha:
            M sin(alpha) + N cos(alpha) = L
            L = |l|^2 - (s^2 - a^2)
            M = 2a l_z
            N = 2a (cos(beta) l_x + sin(beta) l_y)
        Using M sin(alpha) + N cos(alpha) = sqrt(M^2 + N^2) sin(alpha + atan2(N, M)):
            alpha = asin(L / sqrt(M^2 + N^2)) - atan2(N, M)
        If |L / sqrt(M^2 + N^2)| > 1 the rod cannot reach: no real solution.
        The asin branch chosen gives alpha = 0 at home (horn horizontal).
        """
        l = self.leg_vectors(pose)
        big_l = np.einsum("ij,ij->i", l, l) - (self.s ** 2 - self.a ** 2)
        m = 2.0 * self.a * l[:, 2]
        n = 2.0 * self.a * (self.cos_beta * l[:, 0] + self.sin_beta * l[:, 1])
        ratio = big_l / np.hypot(m, n)
        if np.any(np.abs(ratio) > 1.0):
            return None
        return np.arcsin(ratio) - np.arctan2(n, m)

    def solve(self, pose):
        """
        Pose (x, y, z [mm], roll, pitch, yaw [deg]) -> 6 servo commands [deg].

        The pose is clipped to POSE_LIMITS, and the servo commands are clipped
        to [SERVO_MIN_DEG, SERVO_MAX_DEG] so no command can exceed the servo's
        physical range. Returns None if the pose is unreachable; the caller
        should hold the previous command.
        """
        pose = np.clip(np.asarray(pose, float), -self.pose_limits, self.pose_limits)
        alpha = self.horn_angles(pose)
        if alpha is None:
            return None
        cmd = self.servo_neutral + self.servo_direction * np.degrees(alpha)
        return np.clip(cmd, SERVO_MIN_DEG, SERVO_MAX_DEG)


if __name__ == "__main__":
    ik = StewartIK()
    print(f"home height h0 = {ik.h0:.2f} mm")
    print("home      :", np.round(ik.solve([0, 0, 0, 0, 0, 0]), 2))
    print("roll +10  :", np.round(ik.solve([0, 0, 0, 10, 0, 0]), 2))
    print("pitch -8  :", np.round(ik.solve([0, 0, 0, 0, -8, 0]), 2))
    print("z +10     :", np.round(ik.solve([0, 0, 10, 0, 0, 0]), 2))
