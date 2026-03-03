import numpy as np

# Create a rotation matrix from the world frame to the body frame using euler angles

#My comments
#Basically euler angles are yaw, pitch and roll
def euler2rotmat(euler_angles):

    R = np.eye(3) #creates a 3x3 identity matrix

    # Here you need to implement the rotation matrix\
    # First calculate the rotation matrix for each angle (roll, pitch, yaw) cf slide 8 attitude lecture
    # Then multiply the matrices together to get the total rotation matrix
    # Inputs:
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    # Outputs:
    #           R: A 3x3 numpy array that represents the rotation matrix of the euler angles

    # --- YOUR CODE HERE ---

    # R_roll = 
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                       [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])
    
    # R_pitch = 
    R_pitch = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                        [0, 1, 0],
                        [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])
    
    # R_yaw = 
    R_yaw = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                      [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                      [0, 0, 1]])
    # R = R_yaw @ R_pitch @ R_roll
    R = R_yaw @ R_pitch @ R_roll
    return R


# Rotate the control commands from the inertial reference frame to the body reference frame
def rot_inertial2body(control_commands, euler_angles, quaternion):

    # Here you need to rotate the control commands from the inertial reference frame to the body reference frame (du referentiel du drone a celui du monde)
    # You should use the euler2rotmat function to get the rotation matrix
    # Keep in mind that you only want to rotate the velocity commands, which are the first two elements of the control_commands array
    # Think carefully about which direction you need to perform the rotation

    # Inputs:
    #           control_commands: A list of 4 control commands [vel_x, vel_y, altitude, yaw_rate] in the inertial reference frame
    #vel_x      → horizontal velocity along x-axis; vel_y      → horizontal velocity along y-axis;altitude   → vertical position (or vertical velocity depending on controller); yaw_rate   → rotational speed around z-axis
    #these are what allowss to move in the world frame, but we need to convert them to the body frame
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    #           quaternion: A list of 4 elements [x, y, z, w] representing the quaternion of the drone
    # Outputs:
    #           control_commands: A list of 4 control commands [vel_x, vel_y, altitude, yaw_rate] in the body reference frame

    # --- YOUR CODE HERE ---

    # vel_inertial = 
    vel_inertial = np.array([control_commands[0], control_commands[1],0]) #takes the first two elements of control_commands, which are the velocity commands in the inertial frame
    # R = 
    R = euler2rotmat(euler_angles) #gets the rotation matrix from the euler angles
    # vel_body = R inverse * vitesse interielle
    vel_body = R.T @ vel_inertial #rotates the velocity commands from the inertial frame to the body frame using the transpose of the rotation matrix

    # control_commands = 
    control_commands[0] = vel_body[0]
    control_commands[1] = vel_body[1]

    return control_commands