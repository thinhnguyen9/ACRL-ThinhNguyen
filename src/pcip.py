import numpy as np
# from scipy.linalg import cholesky, solve_triangular

class PCIPQP:
    """
    Prediction–Correction Interior Point solver for (unconstrained) Quadratic Program:
        minimize 0.5 z^T H z + f^T z
    """
    def __init__(self, alpha, ts, estimate_grad_zt=False):
        self.alpha   = alpha       # correction gain
        self.tau     = ts          # Euler step
        self.estimate_grad_zt = estimate_grad_zt
    
    def set_QP(self, H, f):
        # preserve previous copies (if present) and store new copies
        # for dH/dt, df/dt estimation using finite differences
        if hasattr(self, 'H'):
            self.H0 = self.H.copy()
        else:
            self.H0 = None
        if hasattr(self, 'f'):
            self.f0 = self.f.copy()
        else:
            self.f0 = None
        # store new QP data as copies to avoid external mutation
        self.H = np.array(H, copy=True)
        self.f = np.array(f, copy=True)

    def dynamics(self, z0):
        grad_phi = self.H @ z0 + self.f
        hess_phi = self.H

        # Estimate grad_zt by finite difference
        if self.estimate_grad_zt \
                and self.H0 is not None and self.f0 is not None \
                and self.H0.shape[0]==self.H.shape[0]:
            g_pred = (grad_phi - self.H0 @ z0 - self.f0)/self.tau
        else:
            g_pred = np.zeros_like(z0)

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
