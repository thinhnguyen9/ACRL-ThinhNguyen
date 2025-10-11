import numpy as np
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ----------------------- Params -----------------------
ts = .01
frame_skip = 5  # Skip every 10 frames
l = .2  # Length of the quadrotor arms
# -------------------------------------------------------

# Load the simulation data from the .npy file
file_path = os.path.join(os.path.dirname(__file__), 'sim_data.npy')
with open(file_path, 'rb') as f:
    tvec = np.load(f)
    xvec = np.load(f)
    xhat = np.load(f)
    yvec = np.load(f)
    uvec = np.load(f) 

# Verify the loaded data
print("tvec shape:", tvec.shape)
print("xvec shape:", xvec.shape)
print("xhat shape:", xhat.shape)
print("yvec shape:", yvec.shape)
print("uvec shape:", uvec.shape)

# Extract position data (x, y, z)
x       = xvec[:, 0]
y       = xvec[:, 2]
z       = xvec[:, 4]
roll    = xvec[:, 6]
pitch   = xvec[:, 7]
yaw     = xvec[:, 8]
x_e     = xhat[:, 0]
y_e     = xhat[:, 2]
z_e     = xhat[:, 4]
roll_e  = xhat[:, 6]
pitch_e = xhat[:, 7]
yaw_e   = xhat[:, 8]

# Reduce the number of frames for real-time animation
tvec = tvec[::frame_skip]
x = x[::frame_skip]
y = y[::frame_skip]
z = z[::frame_skip]
roll = roll[::frame_skip]
pitch = pitch[::frame_skip]
yaw = yaw[::frame_skip]
x_e = x_e[::frame_skip]
y_e = y_e[::frame_skip]
z_e = z_e[::frame_skip]

# Initialize the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Combine actual and estimated trajectories to calculate limits
x_combined = np.concatenate((x, x_e))
y_combined = np.concatenate((y, y_e))
z_combined = np.concatenate((z, z_e))

# Set axis limits and make axes equal
max_range = max(np.ptp(x_combined), np.ptp(y_combined), np.ptp(z_combined)) / 2.0
mid_x = (np.max(x_combined) + np.min(x_combined)) / 2.0
mid_y = (np.max(y_combined) + np.min(y_combined)) / 2.0
mid_z = (np.max(z_combined) + np.min(z_combined)) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Quadrotor Animation')

# Invert the axes (rotate pi by x so z points upwards)
# ax.invert_xaxis()
ax.invert_yaxis()
ax.invert_zaxis()

# Initialize the quadrotor point and trajectory
quadrotor, = ax.plot([], [], [], 'bo', markersize=2, label='Quadrotor')
trajectory, = ax.plot([], [], [], 'r-', lw=1.5, label='Trajectory')
trajectory_est, = ax.plot([], [], [], 'c--', lw=1, label='Estimation')
ax.legend()

# Initialize the quadrotor arms
arm1, = ax.plot([], [], [], 'k-', lw=3)  # Arm 1
arm2, = ax.plot([], [], [], 'k-', lw=3)  # Arm 2

# Initialize the quadrotor body frame axes
body_x_axis, = ax.plot([], [], [], 'r-', lw=.7, label='Body X-axis')  # Red for X-axis
body_y_axis, = ax.plot([], [], [], 'g-', lw=.7, label='Body Y-axis')  # Green for Y-axis
body_z_axis, = ax.plot([], [], [], 'b-', lw=.7, label='Body Z-axis')  # Blue for Z-axis

def rotation_matrix(roll, pitch, yaw):
    """Generate a 3D rotation matrix from roll, pitch, and yaw angles."""
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll), cos(roll)]])
    
    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]])
    
    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw), cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

# Update function for animation
def update(frame):
    # Update quadrotor position
    quadrotor.set_data([x[frame]], [y[frame]])
    quadrotor.set_3d_properties([z[frame]])
    
    # Update trajectory
    trajectory.set_data(x[:frame], y[:frame])
    trajectory.set_3d_properties(z[:frame])
    trajectory_est.set_data(x_e[:frame], y_e[:frame])
    trajectory_est.set_3d_properties(z_e[:frame])
    
    # Calculate arm positions with rotation
    R = rotation_matrix(roll[frame], pitch[frame], yaw[frame])
    arm1_start = np.array([l/sqrt(2), l/sqrt(2), 0])
    arm1_end   = np.array([-l/sqrt(2), -l/sqrt(2), 0])
    arm2_start = np.array([l/sqrt(2), -l/sqrt(2), 0])
    arm2_end   = np.array([-l/sqrt(2), l/sqrt(2), 0])
    
    arm1_start_rotated = R @ arm1_start
    arm1_end_rotated = R @ arm1_end
    arm2_start_rotated = R @ arm2_start
    arm2_end_rotated = R @ arm2_end
    
    arm1_x = [x[frame] + arm1_start_rotated[0], x[frame] + arm1_end_rotated[0]]
    arm1_y = [y[frame] + arm1_start_rotated[1], y[frame] + arm1_end_rotated[1]]
    arm1_z = [z[frame] + arm1_start_rotated[2], z[frame] + arm1_end_rotated[2]]
    
    arm2_x = [x[frame] + arm2_start_rotated[0], x[frame] + arm2_end_rotated[0]]
    arm2_y = [y[frame] + arm2_start_rotated[1], y[frame] + arm2_end_rotated[1]]
    arm2_z = [z[frame] + arm2_start_rotated[2], z[frame] + arm2_end_rotated[2]]
    
    # Update arms
    arm1.set_data(arm1_x, arm1_y)
    arm1.set_3d_properties(arm1_z)
    arm2.set_data(arm2_x, arm2_y)
    arm2.set_3d_properties(arm2_z)
    
    # Update body frame axes
    body_x = R @ np.array([l, 0, 0])
    body_y = R @ np.array([0, l, 0])
    body_z = R @ np.array([0, 0, l])
    
    body_x_axis.set_data([x[frame], x[frame] + body_x[0]], [y[frame], y[frame] + body_x[1]])
    body_x_axis.set_3d_properties([z[frame], z[frame] + body_x[2]])
    
    body_y_axis.set_data([x[frame], x[frame] + body_y[0]], [y[frame], y[frame] + body_y[1]])
    body_y_axis.set_3d_properties([z[frame], z[frame] + body_y[2]])
    
    body_z_axis.set_data([x[frame], x[frame] + body_z[0]], [y[frame], y[frame] + body_z[1]])
    body_z_axis.set_3d_properties([z[frame], z[frame] + body_z[2]])
    
    return quadrotor, trajectory, arm1, arm2, body_x_axis, body_y_axis, body_z_axis

# Create the animation
ani = FuncAnimation(fig, update, frames=len(tvec), interval=ts * 1000 * frame_skip, blit=False)

# Show the animation
plt.show()