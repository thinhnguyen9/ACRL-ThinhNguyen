import numpy as np
# from scipy.linalg import cholesky, solve_triangular

class PCIPQP:
    """
    Prediction–Correction Interior Point solver for (unconstrained) Quadratic Program:
        minimize 0.5 z^T H z + f^T z
    """
    def __init__(self, alpha, ts):
        self.alpha   = alpha       # correction gain
        self.tau     = ts          # Euler step
    
    def set_QP(self, H, f):
        self.H = H
        self.f = f

    def dynamics(self, z0, dH=None, df=None):
        grad_phi = self.H @ z0 + self.f
        hess_phi = self.H

        g_pred = np.zeros_like(z0)
        if dH is not None:
            g_pred += dH @ z0
        if df is not None:
            g_pred += df

        # Solve Newton system
        delta = np.linalg.solve(hess_phi, -self.alpha*grad_phi - g_pred)
        # L = cholesky(hess_phi, lower=True)  # Cholesky decomposition
        # y = solve_triangular(L, -self.alpha*grad_phi - g_pred, lower=True)  # Solve Ly = b
        # delta = solve_triangular(L.T, y)  # Solve L.T x = y
        # delta = np.linalg.inv(H) @ rhs    # slow, not accurate
        # delta = np.zeros_like(z0)

        # Euler integration step
        z = z0 + self.tau*delta
        return delta, z
