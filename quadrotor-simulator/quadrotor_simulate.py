import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import pathlib
from math import sin, cos

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from controllers.controllers import LQR
from models.quadrotors import Quadrotor1

def saturate(val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))

# ----------------------- Sim time -----------------------
ts = 0.01
T = 5.0
tvec = np.arange(0.0, T+ts, ts)
N = len(tvec)

# ----------------------- Quadrotor -----------------------
drone = Quadrotor1(
    l   = .2,
    m   = .8,
    Jx  = .05,
    Jy  = .05,
    Jz  = .07,
    kf  = [0.0, 1.6, 4.0],
    Kdt = [1.99049588e-04, 3.50416237e-02, -1.40556979e-03, 2.61568995e-04, -1.86983501e-05],
    tc  = .1,
    Kdx = 1.,
    Kdy = 1.,
    Kdz = 1.
)
Ny = 9  # can measure x, xd, y, yd, z, zd, roll, pitch, yaw
C = np.zeros((Ny, drone.Nx))
C[0,0] = 1.
C[1,1] = 1.
C[2,2] = 1.
C[3,3] = 1.
C[4,4] = 1.
C[5,5] = 1.
C[6,6] = 1.
C[7,7] = 1.
C[8,8] = 1.

# Measurement noise (gaussian): max error ~ 3 std.dev.
v_means = np.zeros(Ny)
v_stds = np.array([.02, .5, .02, .5, .02, .5, .03, .03, .03])/3

# Process noise (gaussian): max error ~ 3 std.dev.
w_means = np.zeros(drone.Nx)
w_stds = np.zeros(drone.Nx)

# ----------------------- Controller -----------------------
StateFeedback = LQR(
    type = 'continuous',
    n = drone.Nx,
    m = drone.Nu
)
"""
Tune these:
    Q: 1/(maxError^2)
    R: 1/(umax^2)
"""
Q = np.diag([1/(0.05**2), 1/(0.2**2),               # x, xdot
             1/(0.05**2), 1/(0.2**2),               # y, ydot
             1/(0.05**2), 1/(0.2**2),               # z, zdot
             1/(0.1**2), 1/(0.1**2), 1/(0.01**2),   # roll, pitch, yaw
             1/(0.5**2), 1/(0.5**2), 1/(0.5**2),    # angular rates
             0, 0, 0, 0 ])                          # motor thrusts - allow large errors
R = np.diag([1/(drone.umax**2)]*4)
StateFeedback.setWeight(Q, R)

# Linearize around hover position
xhover = np.zeros(drone.Nx)
xhover[12:16] = drone.f_h
uhover = np.array([drone.u_h]*drone.Nu)
A, B = drone.linearize(xhover, uhover)
StateFeedback.setModel(A, B)
StateFeedback.calculateGain()
print("State feedback gain from LQR:")
with np.printoptions(precision=2, suppress=True):
    print(StateFeedback.getGain())

# ----------------------- Estimator -----------------------
KalmanFilter = LQR(
    type = 'continuous',
    n = drone.Nx,
    m = Ny
)
KalmanFilter.setModel(A.T, C.T)
V = np.diag(v_stds)                     # measurement noise covariance
W = np.diag([1., .1, 1., .1, 1., .1,
             1., 1., 1., .1, .1, .1,
             0., 0., 0., 0., ])         # process noise covariance - TODO: tune
KalmanFilter.setWeight(W, V)
KalmanFilter.calculateGain()
print("Estimator gain from Kalman filter:")
with np.printoptions(precision=2, suppress=True):
    print(KalmanFilter.getGain().T)

# ----------------------- x0 & xref -----------------------
x0hat = xhover  # initial estimate

# Initially, hovering at (x,y,z)
x0   = np.array([0., 0., 0., 0., -1., 0.,
                 0., 0., 0.,
                 0., 0., 0.,
                 drone.f_h, drone.f_h, drone.f_h, drone.f_h])

# Goal
xref = np.array([1., 0., .5, 0., -1., 0.,
                 0., 0., 0.,
                 0., 0., 0.,
                 drone.f_h, drone.f_h, drone.f_h, drone.f_h])

# ----------------------- Simulation -----------------------
xvec = np.zeros((N, drone.Nx))
xhat = np.zeros((N, drone.Nx))
yvec = np.zeros((N, Ny))
uvec = np.zeros((N, drone.Nu))
dmax = .2   # bound on x,y,z errors for stability

wvec = np.random.normal(loc=w_means, scale=w_stds, size=(N, drone.Nx))
vvec = np.random.normal(loc=v_means, scale=v_stds, size=(N, Ny))
# wvec = np.zeros((N, drone.Nx))
# vvec = np.zeros((N, Ny))

for i in range(N):
    xvec[i,:] = x0
    yvec[i,:] = C@x0 + vvec[i,:]
    xhat[i,:] = x0hat

    # Bound x,y,z errors for stability
    err = xref - x0
    err[0] = saturate(err[0], -dmax, dmax)
    err[2] = saturate(err[2], -dmax, dmax)
    err[4] = saturate(err[4], -dmax, dmax)

    # u = StateFeedback.getGain()@err + drone.gravityCompensation(x0)
    u = StateFeedback.getGain()@err + uhover
    u = drone.saturateControl(u)
    uvec[i,:] = u

    # solution = solve_ivp(drone.dx, (0,ts), x0, method='RK45', t_eval=[0,ts])
    # t = solution.t
    # X = solution.y

    dx = drone.dx(None, x0, u, wvec[i,:])
    x0 += dx*ts

    # Estimation
    A, B = drone.linearize(x0hat, u)
    KalmanFilter.setModel(A.T, C.T)
    KalmanFilter.calculateGain()
    delta_x = x0hat - xhover
    delta_u = u - uhover
    dxhat = A@delta_x + B@delta_u + KalmanFilter.getGain().T@(yvec[i,:] - C@x0hat)
    # dxhat = drone.dx(None, x0, u, np.zeros(16)) + KalmanFilter.getGain().T@(yvec[i,:] - C@x0hat)
    x0hat += dxhat*ts

# Save simulation data to a .npy file
with open('quadrotor-simulator/sim_data.npy', 'wb') as f:
    np.save(f, tvec)
    np.save(f, xvec)
    np.save(f, xhat)
    np.save(f, yvec)
    np.save(f, uvec)
print("Simulation data saved to 'sim_data.npy'")

# ----------------------- Plot results -----------------------
plt.figure(1)
plt.suptitle('Quadrotor states')

plt.subplot(221)
plt.plot(tvec, xvec[:,0], 'r', lw=1.5, label='x')
plt.plot(tvec, xhat[:,0], 'r--', lw=1, label='x_est')
plt.plot(tvec, xvec[:,2], 'g', lw=1.5, label='y')
plt.plot(tvec, xhat[:,2], 'g--', lw=1, label='y_est')
plt.plot(tvec, xvec[:,4], 'b', lw=1.5, label='z')
plt.plot(tvec, xhat[:,4], 'b--', lw=1, label='z_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position - world (m)')

plt.subplot(222)
plt.plot(tvec, xvec[:,1], 'r', lw=1.5, label='xdot')
plt.plot(tvec, xhat[:,1], 'r--', lw=1, label='xdot_est')
plt.plot(tvec, xvec[:,3], 'g', lw=1.5, label='ydot')
plt.plot(tvec, xhat[:,3], 'g--', lw=1, label='ydot_est')
plt.plot(tvec, xvec[:,5], 'b', lw=1.5, label='zdot')
plt.plot(tvec, xhat[:,5], 'b--', lw=1, label='zdot_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity - world (m/s)')

plt.subplot(223)
plt.plot(tvec, xvec[:,6]*180/np.pi, 'r', lw=1.5, label='roll')
plt.plot(tvec, xhat[:,6]*180/np.pi, 'r--', lw=1, label='roll_est')
plt.plot(tvec, xvec[:,7]*180/np.pi, 'g', lw=1.5, label='pitch')
plt.plot(tvec, xhat[:,7]*180/np.pi, 'g--', lw=1, label='pitch_est')
plt.plot(tvec, xvec[:,8]*180/np.pi, 'b', lw=1.5, label='yaw')
plt.plot(tvec, xhat[:,8]*180/np.pi, 'b--', lw=1, label='yaw_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angles - world (deg)')

plt.subplot(224)
plt.plot(tvec, xvec[:,9]*180/np.pi, 'r', lw=1.5, label='p')
plt.plot(tvec, xhat[:,9]*180/np.pi, 'r--', lw=1, label='p_est')
plt.plot(tvec, xvec[:,10]*180/np.pi, 'g', lw=1.5, label='q')
plt.plot(tvec, xhat[:,10]*180/np.pi, 'g--', lw=1, label='q_est')
plt.plot(tvec, xvec[:,11]*180/np.pi, 'b', lw=1.5, label='r')
plt.plot(tvec, xhat[:,11]*180/np.pi, 'b--', lw=1, label='r_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular rates - body (deg/s)')



plt.figure(2)
plt.suptitle('Kalman filter')

plt.subplot(331)
plt.plot(tvec, xvec[:,0], 'k-', lw=1.5, label='x')
plt.plot(tvec, yvec[:,0], 'k:', lw=0.5, label='x_meas')
plt.plot(tvec, xhat[:,0], 'k--', lw=1., label='x_est')
plt.grid()
plt.legend()
plt.ylabel('Position - world (m)')

plt.subplot(332)
plt.plot(tvec, xvec[:,2], 'k-', lw=1.5, label='y')
plt.plot(tvec, yvec[:,2], 'k:', lw=0.5, label='y_meas')
plt.plot(tvec, xhat[:,2], 'k--', lw=1., label='y_est')
plt.grid()
plt.legend()

plt.subplot(333)
plt.plot(tvec, xvec[:,4], 'k-', lw=1.5, label='z')
plt.plot(tvec, yvec[:,4], 'k:', lw=0.5, label='z_meas')
plt.plot(tvec, xhat[:,4], 'k--', lw=1., label='z_est')
plt.grid()
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(334)
plt.plot(tvec, xvec[:,1], 'k-', lw=1.5, label='xd')
plt.plot(tvec, yvec[:,1], 'k:', lw=0.5, label='xd_meas')
plt.plot(tvec, xhat[:,1], 'k--', lw=1., label='xd_est')
plt.grid()
plt.legend()
plt.ylabel('Velocity - world (m/s)')

plt.subplot(335)
plt.plot(tvec, xvec[:,3], 'k-', lw=1.5, label='yd')
plt.plot(tvec, yvec[:,3], 'k:', lw=0.5, label='yd_meas')
plt.plot(tvec, xhat[:,3], 'k--', lw=1., label='yd_est')
plt.grid()
plt.legend()

plt.subplot(336)
plt.plot(tvec, xvec[:,5], 'k-', lw=1.5, label='zd')
plt.plot(tvec, yvec[:,5], 'k:', lw=0.5, label='zd_meas')
plt.plot(tvec, xhat[:,5], 'k--', lw=1., label='zd_est')
plt.grid()
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(337)
plt.plot(tvec, xvec[:,6]*180/np.pi, 'k-', lw=1.5, label='roll')
plt.plot(tvec, yvec[:,6]*180/np.pi, 'k:', lw=0.5, label='roll_meas')
plt.plot(tvec, xhat[:,6]*180/np.pi, 'k--', lw=1., label='roll_est')
plt.grid()
plt.legend()
plt.ylabel('Angles - world (deg)')
plt.xlabel('Time (s)')

plt.subplot(338)
plt.plot(tvec, xvec[:,7]*180/np.pi, 'k-', lw=1.5, label='pitch')
plt.plot(tvec, yvec[:,7]*180/np.pi, 'k:', lw=0.5, label='pitch_meas')
plt.plot(tvec, xhat[:,7]*180/np.pi, 'k--', lw=1., label='pitch_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')

plt.subplot(339)
plt.plot(tvec, xvec[:,8]*180/np.pi, 'k-', lw=1.5, label='yaw')
plt.plot(tvec, yvec[:,8]*180/np.pi, 'k:', lw=0.5, label='yaw_meas')
plt.plot(tvec, xhat[:,8]*180/np.pi, 'k--', lw=1., label='yaw_est')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')



fig, axes = plt.subplots(2, 2)
fig.suptitle('Control inputs')

ax1 = axes[0,0]
ax2 = ax1.twinx()
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Motor 1 force (N)', c='b')
ax2.set_ylabel('Motor 1 input (%)', c='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.plot(tvec, xvec[:,12], 'b', lw=1.5, label='actual')
ax1.plot(tvec, xhat[:,12], 'b--', lw=.5, label='estimated')
ax2.plot(tvec, uvec[:,0]*100, 'r', lw=.5)
ax1.legend()

ax1 = axes[0,1]
ax2 = ax1.twinx()
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Motor 2 force (N)', c='b')
ax2.set_ylabel('Motor 2 input (%)', c='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.plot(tvec, xvec[:,13], 'b', lw=1.5, label='actual')
ax1.plot(tvec, xhat[:,13], 'b--', lw=.5, label='estimated')
ax2.plot(tvec, uvec[:,1]*100, 'r', lw=.5)
ax1.legend()

ax1 = axes[1,0]
ax2 = ax1.twinx()
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Motor 3 force (N)', c='b')
ax2.set_ylabel('Motor 3 input (%)', c='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.plot(tvec, xvec[:,14], 'b', lw=1.5, label='actual')
ax1.plot(tvec, xhat[:,14], 'b--', lw=.5, label='estimated')
ax2.plot(tvec, uvec[:,2]*100, 'r', lw=.5)
ax1.legend()

ax1 = axes[1,1]
ax2 = ax1.twinx()
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Motor 4 force (N)', c='b')
ax2.set_ylabel('Motor 4 input (%)', c='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.plot(tvec, xvec[:,15], 'b', lw=1.5, label='actual')
ax1.plot(tvec, xhat[:,15], 'b--', lw=.5, label='estimated')
ax2.plot(tvec, uvec[:,3]*100, 'r', lw=.5)
ax1.legend()


plt.show()
