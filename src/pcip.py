import numpy as np
# from scipy.linalg import cholesky, solve_triangular

class PCIPQP:
    """
    Prediction–Correction Interior Point solver for (unconstrained) Quadratic Program:
        minimize 0.5 z^T H z + f^T z
    """
    def __init__(self, alpha, ts, estimate_grad_zt=False):
        self.alpha   = alpha       # correction gain
        self.ts      = ts          # Euler step
        self.estimate_grad_zt = estimate_grad_zt
    
    def set_QP(self, H, f):
        if hasattr(self, 'H') and hasattr(self, 'f'):
            self.H0 = self.H
            self.f0 = self.f
        else:
            self.H0 = H
            self.f0 = f
        self.H = H
        self.f = f

    def dynamics(self, z0):
        grad_phi = self.H @ z0 + self.f
        hess_phi = self.H

        diff = self.H.shape[0] - self.H0.shape[0]
        if diff == 0:   # QP size fixed
            grad_phi0 = self.H0 @ z0 + self.f0
            # hess_phi0 = self.H0
        else:   # QP size grew
            grad_phi0 = np.hstack([
                self.H0 @ z0[:-diff] + self.f0,
                grad_phi[-diff:]
            ])
            # hess_phi0 = self.H.copy()
            # hess_phi0[:-diff, :-diff] = self.H0.copy()

        # Estimate grad_zt_phi by finite difference
        if self.estimate_grad_zt:
            grad_zt = (grad_phi - grad_phi0)/self.ts
        else:
            grad_zt = np.zeros_like(z0)

        # Solution
        zdot = np.linalg.solve(hess_phi, -self.alpha*grad_phi - grad_zt)
        z = z0 + self.ts*zdot  # z(T+1)!!
        return zdot, z
