import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import pathlib
from math import sin, cos
import cvxpy as cp
import copy
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.controllers import LQR
# from src.estimators import KF, MHE
from models.quadrotors import Quadrotor1



class Simulator():

    def __init__(
            self,
            mode,
            sys,
            w_means,
            w_stds,
            v_means,
            v_stds,
            T=5.0,
            ts=0.01,
            noise_distribution="gaussian"
        ):
        if mode not in ['quadrotor', 'reactor']:
            raise ValueError("Invalid mode. Supported modes: 'quadrotor', 'reactor'.")
        if noise_distribution not in ['gaussian', 'uniform']:
            raise ValueError("Invalid noise distribution. Supported distributions: 'gaussian', 'uniform'.")
        self.mode   = mode
        self.sys    = sys
        self.T      = T
        self.ts     = ts
        self.tvec   = np.arange(0.0, T+ts, ts)
        self.N      = len(self.tvec)
        self.Nx     = sys.Nx
        self.Nu     = sys.Nu
        self.Ny     = sys.Ny
        self.w_means    = w_means
        self.w_stds     = w_stds
        self.v_means    = v_means
        self.v_stds     = v_stds
        self.noise_distribution = noise_distribution

        # Set equilibrium point for linearization
        if self.mode == 'quadrotor':
            xhover = np.zeros(sys.Nx)
            xhover[12:16] = sys.f_h
            uhover = np.array([sys.u_h]*sys.Nu)
            self.x_eq = xhover
            self.u_eq = uhover
        else:
            self.x_eq = np.zeros(self.Nx)
            self.u_eq = np.zeros(self.Nu)


    @staticmethod
    def saturate(val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))


    def simulate_quadrotor_lqr_control(self, traj_mode="p2p", x0=None, xref=None, zero_disturbance=False, zero_noise=False):
        """
        traj_mode: "p2p" (point-to-point)
                   "circle" (circular trajectory)
        """
        # ----------------------- Initial & ref states -----------------------
        if self.mode != 'quadrotor':
            return
        else:
            if traj_mode == "p2p":
                if x0 is None:
                    x0 = np.array([ 0., 0., 0., 0., -1., 0.,
                                    0., 0., 0.,
                                    0., 0., 0.,
                                    self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                if xref is None:
                    xref = np.array([ .25, 0., .5, 0., -1., 0.,
                                    0., 0., 0.,
                                    0., 0., 0.,
                                    self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
            
            elif traj_mode == "circle":
                radius = 0.5    # m
                omega = 1.75    # rad/s
                vz = -.1        # m/s
                z0 = -1.
                x0 = np.array([ radius, 0., 0., 0., z0, 0.,
                                0., 0., 0.,
                                0., 0., 0.,
                                self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                xref = np.array([ 0., 0., 0., 0., z0, 0.,
                                0., 0., 0.,
                                0., 0., 0.,
                                self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
        
        # else:
        #     if x0 is None:      x0 = np.zeros(self.Nx)
        #     if xref is None:    xref = np.ones(self.Nx) * .5

        # ----------------------- Controller -----------------------
        StateFeedback = LQR(
            type = 'continuous',
            n = self.Nx,
            m = self.Nu
        )
        """
        Tune these:
            Q: 1/(maxError^2)
            R: 1/(umax^2)
        """
        if self.mode == 'quadrotor':
            Q = np.diag([1/(0.05**2), 1/(0.2**2),               # x, xdot
                        1/(0.05**2), 1/(0.2**2),               # y, ydot
                        1/(0.05**2), 1/(0.2**2),               # z, zdot
                        1/(0.1**2), 1/(0.1**2), 1/(0.01**2),   # roll, pitch, yaw
                        1/(0.5**2), 1/(0.5**2), 1/(0.5**2),    # angular rates
                        0, 0, 0, 0 ])                          # motor thrusts - allow large errors
            R = np.diag([1/(self.sys.umax**2)]*4)
        else:
            Q = np.eye(self.Nx)
            R = np.eye(self.Nu)
        StateFeedback.setWeight(Q, R)

        # Linearize around hover position
        A, B, G, C = self.sys.linearize(self.x_eq, self.u_eq)
        StateFeedback.setModel(A, B)
        StateFeedback.calculateGain()
        dmax = .2   # bound on x,y,z errors for stability

        # ----------------------- Simulation -----------------------
        xvec = np.zeros((self.N, self.Nx))
        yvec = np.zeros((self.N, self.Ny))
        uvec = np.zeros((self.N, self.Nu))
        if zero_disturbance:
            wvec = np.zeros((self.N, self.Nx))
        else:
            if self.noise_distribution=="gaussian":  wvec = np.random.normal(loc=self.w_means, scale=self.w_stds, size=(self.N, self.Nx))
            elif self.noise_distribution=="uniform": wvec = np.random.uniform(low=-self.w_stds*3, high=self.w_stds*3, size=(self.N, self.Nx))
        if zero_noise:
            vvec = np.zeros((self.N, self.Ny))
        else:
            if self.noise_distribution=="gaussian":  vvec = np.random.normal(loc=self.v_means, scale=self.v_stds, size=(self.N, self.Ny))
            elif self.noise_distribution=="uniform": vvec = np.random.uniform(low=-self.v_stds*3, high=self.v_stds*3, size=(self.N, self.Ny))

        for i in range(self.N):
            # if i % 126 == 0:
            #     vvec[i] = vvec[i]*10.   # introduce outliers
            
            if traj_mode=="circle":
                xref[0] = radius * cos(omega*self.tvec[i])
                xref[2] = radius * sin(omega*self.tvec[i])
                xref[4] = z0 + vz * self.tvec[i]
                xref[1] = -radius * omega * sin(omega*self.tvec[i])
                xref[3] = radius * omega * cos(omega*self.tvec[i])
                xref[5] = vz

            xvec[i,:] = x0
            yvec[i,:] = self.sys.getOutput(x0, vvec[i,:])

            # Bound x,y,z errors for stability
            err = xref - x0
            err[0] = self.saturate(err[0], -dmax, dmax)
            err[2] = self.saturate(err[2], -dmax, dmax)
            err[4] = self.saturate(err[4], -dmax, dmax)

            # Calculate control input
            u = StateFeedback.getGain()@err + self.u_eq
            u = self.sys.saturateControl(u)
            uvec[i,:] = u

            # Propagate dynamics
            dx = self.sys.dx(x0, u, wvec[i,:])
            x0 += dx*self.ts

        self.xvec = xvec
        self.uvec = uvec
        self.yvec = yvec
        return self.tvec, xvec, uvec, yvec
    

    def simulate_free_response(self, x0, u=None, zero_disturbance=False, zero_noise=False):
        xvec = np.zeros((self.N, self.Nx))
        yvec = np.zeros((self.N, self.Ny))
        if u is None:
            uvec = np.zeros((self.N, self.Nu))
        else:
            uvec = u.reshape((self.N, self.Nu))
        if zero_disturbance:
            wvec = np.zeros((self.N, self.Nx))
        else:
            if self.noise_distribution=="gaussian":  wvec = np.random.normal(loc=self.w_means, scale=self.w_stds, size=(self.N, self.Nx))
            elif self.noise_distribution=="uniform": wvec = np.random.uniform(low=-self.w_stds*3, high=self.w_stds*3, size=(self.N, self.Nx))
        if zero_noise:
            vvec = np.zeros((self.N, self.Ny))
        else:
            if self.noise_distribution=="gaussian":  vvec = np.random.normal(loc=self.v_means, scale=self.v_stds, size=(self.N, self.Ny))
            elif self.noise_distribution=="uniform": vvec = np.random.uniform(low=-self.v_stds*3, high=self.v_stds*3, size=(self.N, self.Ny))

        for i in range(self.N):
            xvec[i,:] = x0
            yvec[i,:] = self.sys.getOutput(x0, vvec[i,:])
            dx = self.sys.dx(x0, uvec[i,:], wvec[i,:])
            x0 += dx*self.ts

        self.xvec = xvec
        self.uvec = uvec
        self.yvec = yvec
        return self.tvec, xvec, uvec, yvec

    
    def run_estimation(self, estimator, x0_est):
        x0hat = copy.deepcopy(x0_est)       # Initial estimate
        xhat = np.zeros((self.N, self.Nx))  # All estimates
        estimator_class = type(estimator).__name__
        if estimator_class not in ['KF', 'MHE']:
            raise ValueError("Invalid estimator class. Supported classes: KF, MHE.")

        t0 = time.perf_counter()
        for i in range(self.N):
            # -------------- Kalman filter --------------
            if estimator_class == 'KF':
                x0hat = estimator.correction(x0hat, self.yvec[i])
                xhat[i] = x0hat
                x0hat = estimator.prediction(x0hat, self.uvec[i])

            # -------------- Linear MHE --------------
            elif estimator_class == 'MHE':
                x0hat = estimator.doEstimation(
                    yvec = self.yvec[: i+1],
                    uvec = self.uvec[: i]
                )
                xhat[i] = x0hat
                # estimator.updateCovariance(xhat[i], self.uvec[i])
        elapsed_time = time.perf_counter() - t0
        return xhat, elapsed_time


    def get_time_step(self):
        return self.ts
