import math

base_radius_to_horn = 100.0 # centre to servo shaft in mm
horn_length = 25.0 # horn length
rod_length = 150.0 # rod length
offset = 90.0
direction = 1

platform_radius = math.sqrt(base_radius_to_horn**2 + horn_length**2)
delta = math.atan2(horn_length, base_radius_to_horn)  # platform joints sit this far from each servo

Servo_theta = []
Servo_pos = []
Platform_pos = []
horn_angles = []
for i in range(6):
    theta = math.radians(60 * i)
    curent_servo = [base_radius_to_horn * math.cos(theta), base_radius_to_horn * math.sin(theta), 0]
    Servo_pos.append(curent_servo)
    Servo_theta.append(theta)
    horn_angles.append(theta + math.pi / 2)
    Platform_pos.append([platform_radius * math.cos(theta + delta), platform_radius * math.sin(theta + delta), 0])


# RL model outputs a translation (x, y, z) and rotation angles (roll, pitch, yaw)
def inverse_kinematics(x, y, z, roll_deg, pitch_deg, yaw_deg):
    phi, th, psi = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(th), math.sin(th)
    cp, sp = math.cos(psi), math.sin(psi)
    Rotation_matrix = [[cp*ct, cp*st*sf - sp*cf, cp*st*cf + sp*sf],
         [sp*ct, sp*st*sf + cp*cf, sp*st*cf - cp*sf],
         [-st,   ct*sf,            ct*cf]]
    T = [x, y, rod_length + z]  # translation

    cmds = []
    for k in range(6):
        P = Platform_pos[k]
        B = Servo_pos[k]
        Pk = []
        for i in range(3):
            Pk.append(T[i] + Rotation_matrix[i][0]*P[0] + Rotation_matrix[i][1]*P[1] + Rotation_matrix[i][2]*P[2])
        leg_vector_x, leg_vector_y, leg_vector_z = Pk[0] - B[0], Pk[1] - B[1], Pk[2] - B[2]

        e = 2 * horn_length * leg_vector_z  # Vertical
        f = 2 * horn_length * (math.cos(horn_angles[k]) * leg_vector_x + math.sin(horn_angles[k]) * leg_vector_y)  # Horizontal
        g = (leg_vector_x**2 + leg_vector_y**2 + leg_vector_z**2) - (rod_length**2 - horn_length**2)  # Magnitude

        rho = math.sqrt(e**2 + f**2)
        if abs(g / rho) > 1.0:
            return None
        alpha = math.asin(g / rho) - math.atan2(f, e)

        alpha_degres = math.degrees(alpha)

        cmd = offset + direction * alpha_degres
        cmds.append(min(180.0, max(0.0, cmd)))
    return cmds
