import numpy as np
from math import sin, cos   
import cvxpy as cp
from scipy.optimize import minimize, LinearConstraint
import copy


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
            # assume y=h(x) and does not depend on u - choose random u
            _, _, _, C = self.model.linearize(x0, np.zeros(self.Nu))

        # L(k) = P0(k)*C(k)'*inv(R + C(k)*P0(k)*C(k)')
        L = self.P0 @ C.T @ np.linalg.inv(self.R + C @ self.P0 @ C.T)

        # P(k) = P0(k) - L(k)*C(k)*P0(k)
        self.P = self.P0 - L @ C @ self.P0

        # correction: x(k) = x0(k) + L(k)*(y(k) - h(x0(k)))
        x = x0 + L@(y - C@x0)
        return x



class MHE():

    def __init__(self, model, ts, Q, R, N, X0, P0, xs, us, mhe_type="linearized_once", mhe_update="filtering", prior_method="zero"):
        """
        Args:
            model: dynamical model object
            ts: sampling time (for discretization)
            Q: process noise covariance
            R: measurement noise covariance
            N: prediction horizon
            X0: mean of initial state (shape: (Nx,))
            P0: covariance of initial state (shape: (Nx, Nx))
            xs, us: linearization point
            mhe_type: "linearized_once" to linearize at xs, us,
                      "linearized_every" to linearize at each step,
                      "nonlinear" to use the nonlinear dynamics (very slow)
            mhe_update: "filtering" use x(T-N|T-N), i.e. do not override xvec
                        "smoothing" use x(T-N|T-1), i.e. override xvec with latest estimate
            prior_method: "zero" to use zero prior weighting,
                          "ekf" to use the EKF covariance update,
                          "uniform" to use a fixed prior weighting P0
        """
        self.model = model
        self.Nx = model.Nx  # states
        self.Nu = model.Nu  # inputs
        self.Ny = model.Ny  # outputs
        self.ts = ts        # sampling time
        self.Q = Q      # process noise covariance
        self.R = R      # measurement noise covariance
        self.N = N      # prediction horizon
        self.Q_inv = np.linalg.inv(self.Q)
        self.R_inv = np.linalg.inv(self.R)
        
        self.xs = xs
        self.us = us
        if mhe_type not in ["linearized_once", "linearized_every", "nonlinear"]:
            raise ValueError("mhe_type must be 'linearized_once', 'linearized_every', or 'nonlinear'.")
        if mhe_update not in ["filtering", "smoothing"]:
            raise ValueError("mhe_update must be 'filtering' or 'smoothing'.")
        if prior_method not in ["zero", "ekf", "uniform"]:
            raise ValueError("prior_method must be 'zero', 'ekf', or 'uniform'.")
        self.mhe_type = mhe_type
        self.mhe_update = mhe_update
        self.prior_method = prior_method
        A, B, G, C = self.model.linearize(xs, us)
        self.updateModel(A, B, G, C)
        
        self.xvec = np.zeros((1, self.Nx))              # estimates  x(T-N)...x(T) - len: N+1
                                                        # filtering scheme: x(T-N|T-N),...,x(T-1|T-1),x(T|T)
                                                        # smoothing scheme: x(T-N|T),...,x(T-1|T),x(T|T)
        self.Pvec = np.zeros((1, self.Nx, self.Nx))     # covariance P(k|k-1):  P(T-N)...P(T) - len: N+1
        self.Pvec1 = np.zeros((1, self.Nx, self.Nx))    # covariance P(k|k):    P(T-N)...P(T) - len: N+1
        self.xvec[0] = X0
        self.Pvec[0] = P0
        self.Pvec1[0] = P0
        self.P0 = P0

    def updateModel(self, A, B, G, C):
        # Discretize
        self.A = np.eye(self.Nx) + A*self.ts
        self.B = B*self.ts
        self.G = G*self.ts
        self.C = C

    def doEstimation(self, yvec, uvec):
        """
        Run MHE to estimate state trajectory over the horizon.
        
        Args:
            yvec: sequence of outputs y(0)...y(T)
            uvec: sequence of inputs u(0)...u(T-1)
        Returns:
            Estimated current state x(T)
        """
        # ----------------- Define horizon -----------------
        T = np.size(yvec, 0) - 1
        N = min(self.N, T)
        if np.size(uvec, 0) != T:
            raise ValueError("yvec and uvec did not agree in size (yvec must have N+1 rows, uvec must have N rows)!")
        
        if T < self.N:
            # Do Full Information Estimation (FIE) if T < N
            # self.xvec  : x(0)...x(T-1)
            # self.Pvec  : P(0)...P(T|T-1)
            # self.Pvec1 : P(0)...P(T-1|T-1)
            X0 = self.xvec[0]   # x(0)
            if self.prior_method == "zero":         pass
            elif self.prior_method == "uniform":    P0 = self.P0
            elif self.prior_method == "ekf":        P0 = self.Pvec[0]   # P(0)
            yseq_raw = yvec    # y(0)...y(T)
            useq_raw = uvec    # u(0)...u(T-1)
        
        else:
            # self.xvec  : x(T-N-1)...x(T-1)
            # self.Pvec  : P(T-N|T-N-1)...P(T|T-1)
            # self.Pvec1 : P(T-N-1|T-1)...P(T-1|T-1)
            X0 = self.xvec[1]   # x(T-N)
            if self.prior_method == "zero":         pass
            elif self.prior_method == "uniform":    P0 = self.P0
            elif self.prior_method == "ekf":        P0 = self.Pvec[0]   # P(T-N|T-N-1)
            yseq_raw = yvec[-self.N-1 :]   # y(T-N)...y(T)
            useq_raw = uvec[-self.N :]     # u(T-N)...u(T-1)

        # TODO: Backward interation to find P(T-1|T-1)...P(T-N|T-1) for smoothing scheme
        # RAUCH, TUNG and STRIEBEL, 1965
        if self.mhe_update == "smoothing" and N > 1:
            # self.xvec  : x(T-N-1)...x(T-1)            (len: N+1)
            # self.Pvec  : P(T-N|T-N-1)...P(T|T-1)      (len: N+1)
            # self.Pvec1 : P(T-N-1|T-1)...P(T-1|T-1)    (len: N)
            # useq_raw   : u(T-N)...u(T-1)              (len: N)

            # given P(T-1|T-1), iterate from P(T-2|T-1) till P(T-N|T-1) (N-1 steps)
            P_temp = self.Pvec1[-1]    # P(T-1|T-1)
            for i in range(N-1):    # k=T-2,...,T-N
                A, _, _, _ = self.model.linearize(self.xvec[-i-2], useq_raw[-i-2])  # A(T-2)
                A = np.eye(self.Nx) + A*self.ts

                # C(k) = P(k|k) * A'(k) * inv(P(k+1|k))
                try:    C = self.Pvec1[-i-2] @ A.T @ np.linalg.inv(self.Pvec[-i-2]) # C(T-2) (NOT output matrix)
                except: C = self.Pvec1[-i-2] @ A.T @ np.linalg.pinv(self.Pvec[-i-2])

                # P(k|T-1) = P(k|k) + C(k)(P(k+1|T-1) - P(k+1|k))C'(k)
                # start: k=T-2, end: k=T-N
                P_temp = self.Pvec1[-i-2] + C @ (P_temp - self.Pvec[-i-2]) @ C.T    # P(T-2|T-1)
            P0 = P_temp
        
            # adjust arrival cost for y(T-N)...y(T-1) (len: N)
            O = np.zeros((self.Ny*N, self.Nx*N))
            # TODO: wtf the matrix O how?? A,C are time-varying
            for i in range(1, N):       # block row
                for j in range(1, i+1): # block col (only lower-triangular part)
                    # iterate T-N+1...T-1
                    A, _, _, C = self.model.linearize(self.xvec[-N+i-j], useq_raw[-N+i-j])  # TODO: no fcking idea what i'm doing
                    A = np.eye(self.Nx) + A*self.ts
                    O[i*self.Ny:(i+1)*self.Ny, j*self.Nx:(j+1)*self.Nx] = C @ np.linalg.matrix_power(A, i-j)
            W = np.kron(np.eye(N), self.R) + O @ np.kron(np.eye(N), self.Q) @ O.T   # shape: (N*Ny, N*Ny)
            try:    W_inv = np.linalg.inv(W)
            except: W_inv = np.linalg.pinv(W)
            # W_inv = np.eye(self.Ny*N)*0.
            
            # See Rao 2001
            O = np.zeros((self.Ny*N, self.Nx))
            for k in range(N):  # iterate T-N...T-1
                A, _, _, C = self.model.linearize(self.xvec[-N+k], useq_raw[-N+k])
                A = np.eye(self.Nx) + A*self.ts
                O[k*self.Ny : (k+1)*self.Ny, :] = C @ np.linalg.matrix_power(A, k)

        # ----------------- Linearization -----------------
        if self.mhe_type == "linearized_once":
            X0 = X0 - self.xs
            y = yseq_raw - self.C @ self.xs
            u = useq_raw - self.us

        elif self.mhe_type == "linearized_every":
            # Use nonlinear model to get nominal trajectory
            xnom = np.zeros((N+1, self.Nx))     # x(T-N)...x(T)
            xnom[0] = X0
            for k in range(N):
                xnom[k+1] = xnom[k] + self.model.dx(xnom[k], useq_raw[k])*self.ts
                
            X0 = np.zeros(self.Nx)
            y = yseq_raw - xnom @ self.C.T
            u = np.zeros((N, self.Nu))

        elif self.mhe_type == "nonlinear":
            # X0 = X0
            y = yseq_raw
            u = useq_raw

        # ----------------- Optimization - linear MHE (CVXPY) -----------------
        if self.mhe_type in ["linearized_once", "linearized_every"]:

            # Variables
            x = cp.Variable((N+1, self.Nx))         # state
            if N>0: w = cp.Variable((N, self.Nx))   # process noise

            # Calculate cost from T-N to T-1 (N steps)
            # Arrival cost - adjusted for smoothing scheme
            if self.prior_method == "zero":
                cost = 0.0
            else:
                cost = .5*cp.quad_form(x[0] - X0, np.linalg.inv(P0))
            if self.mhe_update == "smoothing" and N > 1:
                yflat = y[:-1].flatten()            # y(T-N)...y(T-1)
                # cost -= .5*cp.quad_form(yflat - O@x[0], W_inv)  # TODO: nonconvex problem!!
            constraints = []
            if self.mhe_type == "linearized_once":      x_cons = self.model.stateConstraint(x[0] + self.xs)
            elif self.mhe_type == "linearized_every":   x_cons = self.model.stateConstraint(x[0] + xnom[0])
            if x_cons is not None:
                constraints.append(x_cons)
            
            # Running cost
            for k in range(N):
                if self.mhe_type == "linearized_every":   # Linearize around nominal trajectory
                    A, B, G, C = self.model.linearize(xnom[k], useq_raw[k])
                    self.updateModel(A, B, G, C)
                cost += .5*cp.quad_form(w[k], self.Q_inv) + .5*cp.quad_form(y[k] - self.C@x[k], self.R_inv)
                constraints.append(x[k+1] == self.A@x[k] + self.B@u[k] + self.G@w[k])
                if self.mhe_type == "linearized_once":      x_cons = self.model.stateConstraint(x[k+1] + self.xs)
                elif self.mhe_type == "linearized_every":   x_cons = self.model.stateConstraint(x[k+1] + xnom[k+1])
                if x_cons is not None:
                    constraints.append(x_cons)
            
            # Calculate cost at time T
            if self.mhe_type == "linearized_every":
                A, B, G, C = self.model.linearize(xnom[-1], np.zeros(self.Nu))   # only use C=dh(x)/dx so u doesn't matter
                self.updateModel(A, B, G, C)
            cost += .5*cp.quad_form(y[-1] - self.C@x[-1], self.R_inv)

            prob = cp.Problem(cp.Minimize(cost), constraints)
            # prob.solve(solver=cp.OSQP, warm_start=True)
            # prob.solve(solver=cp.ECOS)
            try:
                prob.solve()
            except:
                prob.solve(solver=cp.ECOS)

            # Result
            if self.mhe_type == "linearized_once":
                xvec = x.value + self.xs   # x(T-N)...x(T)
            elif self.mhe_type == "linearized_every":
                xvec = x.value + xnom      # x(T-N)...x(T)

        # ----------------- Optimization - nonlinear MHE (scipy.optimize) -----------------
        elif self.mhe_type in ["nonlinear"]:
            def cost_fun(z):    # for nonlinear MHE using scipy.optimize.minimze
                x0 = z[ : self.Nx]
                w = z[self.Nx : ].reshape((N, self.Nx))

                # Arrival cost - adjusted for smoothing scheme
                if self.prior_method == "zero":
                    cost = 0.0
                else:
                    cost = .5 * (x0 - X0).T @ np.linalg.inv(P0) @ (x0 - X0)
                if self.mhe_update == "smoothing" and N > 1:
                    # a_random_matrix = np.zeros((self.Ny*N, self.Nu*N))
                    # for r in range(N):
                    #     for c in range(r):
                    #         a_random_matrix[r*self.Ny:(r+1)*self.Ny, c*self.Nx:(c+1)*self.Nx] = self.C @ np.linalg.matrix_power(self.A, r-c-1) @ self.B
                    # uflat = u.flatten()         # u(T-N)...u(T-1)
                    # yflat = y[:-1].flatten()    # y(T-N)...y(T-1)
                    # temp = yflat - O@x0 - a_random_matrix@uflat
                    # cost -= .5 * temp.T @ W_inv @ temp

                    yflat = y[:-1].flatten()    # y(T-N)...y(T-1)
                    temp = yflat - O@x0
                    cost -= .5 * temp.T @ W_inv @ temp

                # Running cost
                for k in range(N):
                    y_pred = self.model.getOutput(x0)
                    cost += .5*w[k].T @ self.Q_inv @ w[k] + .5*(y[k] - y_pred).T @ self.R_inv @ (y[k] - y_pred)
                    x0 = x0 + self.model.dx(x0, u[k], w[k]) * self.ts   # x(k+1)
                y_pred = self.model.getOutput(x0)
                cost += .5*(y[N] - y_pred).T @ self.R_inv @ (y[N] - y_pred)
                return cost
            
            state_constraint = []
            # A = np.zeros([2, self.Nx + N*self.Nx])
            # A[0,0] = 1.
            # A[1,1] = 1.
            # state_constraint = LinearConstraint(A, 0., np.inf)
        
            w_init = np.zeros(self.Nx*N)
            z_init = np.concatenate([X0, w_init])
            res = minimize(cost_fun, z_init, constraints=state_constraint, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
            x0 = res.x[ : self.Nx]
            w = res.x[self.Nx : ].reshape((N, self.Nx))
            # Rescontruct trajectory
            xvec = np.zeros((N+1, self.Nx))
            xvec[0] = x0
            for k in range(N):
                xvec[k+1] = xvec[k] + self.model.dx(xvec[k], u[k], w[k]) * self.ts
            # print("Done 1 loop of nonlinear MHE!")

        # ----------------- Update self.xvec -----------------
        # self.xvec:    x(T-N-1)...x(T-1)
        #      xvec:    x(T-N)...x(T)
        if self.mhe_update == "filtering":
            # if T > 0:
                # Only save the latest estimate x(T|T)
                self.xvec = np.concatenate((self.xvec, xvec[-1].reshape(1, self.Nx)), axis=0)
                if np.size(self.xvec, 0) > self.N+1:
                    self.xvec = self.xvec[-self.N-1:]
            # else:
            #     self.xvec = xvec    # only 1 value, override initial guess X0
        elif self.mhe_update == "smoothing":    # always override even at T=0,1 - trust me bro
            # Save the entire horizon of latest estimate x(T-N|T)...x(T|T)
            self.xvec = xvec

        # ----------------- Update covariance -----------------
        if self.prior_method == "ekf":

            # Calculate P(T|T) from P(T|T-1)
            P0 = self.Pvec[-1]  # P(T|T-1)
            if self.mhe_type in ["linearized_every", "nonlinear"]:  # Linearize around xhat(T|T)
                A, B, G, C = self.model.linearize(self.xvec[-1], useq_raw[-1] if N > 0 else self.us)    # TODO: do we need correct u here?
                self.updateModel(A, B, G, C)
            L = P0 @ self.C.T @ np.linalg.inv(self.R + self.C @ P0 @ self.C.T)
            P = P0 - L @ self.C @ P0    # P(T|T)
            self.Pvec1 = np.concatenate((self.Pvec1, P.reshape((1, self.Nx, self.Nx))), axis=0)
            if np.size(self.Pvec1, 0) > self.N+1:
                self.Pvec1 = self.Pvec1[-self.N-1:]

            # Calculate P(T+1|T) from P(T|T)
            P = self.G @ self.Q @ self.G.T + self.A @ P @ self.A.T  # P(T+1|T)
            self.Pvec = np.concatenate((self.Pvec, P.reshape((1, self.Nx, self.Nx))), axis=0)
            if np.size(self.Pvec, 0) > self.N+1:
                self.Pvec = self.Pvec[-self.N-1:]

            # if self.mhe_update == "filtering":
            #     # ============================================
            #     #    Gives almost the same result as EKF
            #     # ============================================
                
            #     # self.Pvec: P(T-N|T-N)...P(T-1|T-1),P(T|T)
            #     if self.mhe_type in ["linearized_every", "nonlinear"]:
            #         # Linearize around the last state estimate
            #         A, B, G, C = self.model.linearize(self.xvec[-1], useq_raw[-1] if N > 0 else self.us)    # TODO: do we need correct u here?
            #         self.updateModel(A, B, G, C)
            #     # P0 = self.Pvec[-1]  # P(T)
            #     # P = self.G @ self.Q @ self.G.T \
            #     #     + self.A @ P0 @ self.A.T \
            #     #     - self.A @ P0 @ self.C.T @ np.linalg.inv(self.C @ P0 @ self.C.T + self.R) @ self.C @ P0 @ self.A.T
            #     P = self.G @ self.Q @ self.G.T + self.A @ P0 @ self.A.T
            #     self.Pvec = np.concatenate((self.Pvec, P.reshape((1, self.Nx, self.Nx))), axis=0)
            #     # self.Pvec = self.Pvec[-N-1:]  # P(T-N+1)...P(T+1)
            #     if np.size(self.Pvec, 0) > self.N+1:
            #         self.Pvec = self.Pvec[-self.N-1:]
                
            #     # ============================================

            #     # ============================================
            #     #    Calculate entire Pvec horizon again from fixed P(T-N)
            #     # ============================================
            #     """
            #     #        P0:    P(T-N)
            #     # self.Pvec:    P(T-N)...P(T)
            #     for k in range(N+1):
            #         self.Pvec[k] = P0
            #         if self.mhe_type in ["linearized_every", "nonlinear"]:
            #             A, B, G, C = self.model.linearize(self.xvec[k], self.us)  # TODO: do we need correct u here?
            #             self.updateModel(A, B, G, C)
            #         # P(k+1)
            #         P0 = self.G @ self.Q @ self.G.T \
            #             + self.A @ P0 @ self.A.T \
            #             - self.A @ P0 @ self.C.T @ np.linalg.inv(self.C @ P0 @ self.C.T + self.R) @ self.C @ P0 @ self.A.T
            #     self.Pvec = np.concatenate((self.Pvec, P0.reshape((1, self.Nx, self.Nx))), axis=0)
            #     if np.size(self.Pvec, 0) > self.N+1:
            #         self.Pvec = self.Pvec[-self.N-1:]
            #     """
            
            # elif self.mhe_update == "smoothing":
                
            #     # Obtain P(T|T) given P(T|T-1)
            #     P0 = self.Pvec[-1]  # P(T|T-1)
            #     _, _, _, C = self.model.linearize(self.xvec[-1], self.us)   # TODO: need correct u(T) here??
            #     L = P0 @ C.T @ np.linalg.inv(self.R + C @ P0 @ C.T)
            #     self.Pvec[-1] = P0 - L @ C @ P0

            #     # Obtain P(T+1|T) given P(T|T)
            #     if self.mhe_type in ["linearized_every", "nonlinear"]:
            #         # Linearize around the last state estimate
            #         A, B, G, C = self.model.linearize(self.xvec[-1], useq_raw[-1] if N > 0 else self.us)    # TODO: do we need correct u here?
            #         self.updateModel(A, B, G, C)
            #     P0 = self.Pvec[-1]  # P(T)
            #     P = self.G @ self.Q @ self.G.T + self.A @ P0 @ self.A.T
            #     self.Pvec = np.concatenate((self.Pvec, P.reshape((1, self.Nx, self.Nx))), axis=0)
            #     # self.Pvec = self.Pvec[-N-1:]  # P(T-N+1)...P(T+1)
            #     if np.size(self.Pvec, 0) > self.N+1:
            #         self.Pvec = self.Pvec[-self.N-1:]

            #     """
            #     # self.Pvec: P(T-N|T)...P(T-1|T),P(T|T)
            #     for k in range(N+1):
            #         self.Pvec[k] = P0
            #         if self.mhe_type in ["linearized_every", "nonlinear"]:
            #             A, B, G, C = self.model.linearize(self.xvec[k], self.us)  # TODO: do we need correct u here?
            #             self.updateModel(A, B, G, C)
            #         # P(k+1)
            #         P0 = self.G @ self.Q @ self.G.T \
            #             + self.A @ P0 @ self.A.T \
            #             - self.A @ P0 @ self.C.T @ np.linalg.inv(self.C @ P0 @ self.C.T + self.R) @ self.C @ P0 @ self.A.T
            #     self.Pvec = np.concatenate((self.Pvec, P0.reshape((1, self.Nx, self.Nx))), axis=0)
            #     if np.size(self.Pvec, 0) > self.N+1:
            #         self.Pvec = self.Pvec[-self.N-1:]
            #     """

        # ----------------- Done -----------------
        # self.uvec = u
        # return self.xvec
        return self.xvec[-1]      # x(T)
    
