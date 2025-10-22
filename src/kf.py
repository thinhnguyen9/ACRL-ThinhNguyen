import numpy as np


class KF():

    def __init__(self, model, ts, Q, R, P0=None, type="standard", xs=None, us=None):
        """
        Args:
            model: dynamical model object
            ts: sampling time (for discretization)
            Q: process noise covariance
            R: measurement noise covariance
            P0: initial covariance of x0hat before measurement
            type: "standard" or "extended" Kalman filter
            xs, us: linearization point, only for "standard" Kalman filter
        """
        self.model = model
        self.Nx = model.Nx  # states
        self.Nu = model.Nu  # inputs
        self.Ny = model.Ny  # outputs
        self.ts = ts        # sampling time
        self.Q = Q      # process noise covariance
        self.R = R      # measurement noise covariance
        if P0 is not None:
            self.P0 = P0    # covariance of x0hat before measurement
            self.P = P0     # covariance of x0hat after measurement
        else:
            self.P0 = np.eye(self.Nx)
            self.P = np.eye(self.Nx)
        self.type = type
        if self.type == "standard":
            if xs is None or us is None:
                raise ValueError("For 'standard' Kalman filter, xs and us must be provided.")
            self.xs, self.us = xs, us
            self.A, self.B, self.G, self.C = self.model.linearize(xs, us)

    def update_covariance(self, Q, R):
        """
        In case of time-varying Q, R matrices.
        """
        self.Q = Q
        self.R = R

    def prediction(self, x0, u0):
        """
        Predict the mean and covariance of xhat(k+1) before correction.
        
        Args:
            x0: xhat at time k (after correction)
            u0: control input at time k
        Returns:
            x: mean of xhat at time k+1 (before correction)
        """
        # x0(k+1)
        if self.type == "standard":
            dx = self.A @ (x0 - self.xs) + self.B @ (u0 - self.us)
        elif self.type == "extended":
            dx = self.model.dx(x0, u0)
        x = x0 + dx*self.ts

        # P0(k+1) = A(k)*P(k)*A(k)' + G(k)*Q*G(k)'
        if self.type == "standard":
            A = self.A
            G = self.G
        elif self.type == "extended":
            # A, G linearized around xhat(k) (after correction)
            A, _, G, _ = self.model.linearize(x0, u0)
        A = np.eye(self.Nx) + A*self.ts
        G = G*self.ts
        self.P0 = A @ self.P @ A.T + G @ self.Q @ G.T

        return x
        
    def correction(self, x0, y):
        """
        Update the mean and covariance of xhat(k) after measurement.
        
        Args:
            x0: mean of xhat(k) before correction
            y: measurement at time k
        Returns:
            x: mean of xhat(k) after correction
        """
        if self.type == "standard":
            C = self.C
        elif self.type == "extended":
            # C linearized around xhat(k) (before correction)
            # assume y=h(x) and does not depend on u - choose random u - TODO: need latest u?
            _, _, _, C = self.model.linearize(x0, np.zeros(self.Nu))

        # L(k) = P0(k)*C(k)'*inv(R + C(k)*P0(k)*C(k)')
        L = self.P0 @ C.T @ np.linalg.inv(self.R + C @ self.P0 @ C.T)

        # P(k) = P0(k) - L(k)*C(k)*P0(k)
        self.P = self.P0 - L @ C @ self.P0

        # correction: x(k) = x0(k) + L(k)*(y(k) - h(x0(k)))
        x = x0 + L@(y - C@x0)
        return x
