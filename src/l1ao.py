import numpy as np

class L1AOQP():
    """
    L1 Adaptive Optimization solver for (unconstrained) Quadratic Program:
        minimize 0.5 z^T H z + f^T z
    """
    def __init__(self, ts, a, lpf_omega, estimate_grad_zt=False):
        self.ts = ts
        self.estimate_grad_zt = estimate_grad_zt

        # As: diagonal Hurwitz matrix (assume As = diag([a, a, ..., a]), a<0)
        self.a = a
        self.dim = 0

        # Low-pass filter
        self.lpf_omega = lpf_omega

    def dimension_update(self, dim):
        """
        mu = inv(inv(As)*(I - expm(As*Ts)))*expm(As*Ts)
        Below implementation is only true for diagonal As
        For the MHE problem: dim is continuously growing until it reaches the horizon length
        """
        self.As = np.diag([self.a]*dim)
        self.mu = np.diag([self.a / (np.exp(-self.a*self.ts) - 1.)]*dim)
        self.dim = dim

    def lpf(self, x0, u):
        """
        C(s) = omega / (s + omega)
        xdot = -omega*x + omega*u
        """
        return x0 + self.lpf_omega*(u - x0)*self.ts

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
    
    def dynamics(self, z0, za_dot0, grad_phi_hat0, zb_dot):
        """
        Assumes all vectors are of correct dimension (w.r.t. self.H, self.f). We cannot save these last values
        in this class because as the dimension grows, the previous values must be updated depending upon MHE
        formulation. This is handled in "mhe.py".

        Args:
            z0 := z(T-1): last solution
            za_dot0 := za_dot(T-1): L1AO derivative from last time step
            grad_phi_hat0 := grad_phi_hat(T-1): gradient prediction from last time step
            zb_dot(T): from baseline TV optimizer, e.g. PCIP
        """
        Nz = z0.shape[0]
        if Nz != self.dim:
            self.dimension_update(Nz)   # Update As, mu
        grad_phi = self.H @ z0 + self.f
        hess_phi = self.H

        # Estimate grad_zt by finite difference
        if self.estimate_grad_zt \
                and self.H0 is not None and self.f0 is not None \
                and self.H0.shape[0]==self.H.shape[0]:
            grad_zt = (grad_phi - self.H0 @ z0 - self.f0)/self.ts
        else:
            grad_zt = np.zeros_like(z0)

        # Gradient predictor
        e = grad_phi_hat0 - grad_phi
        h = self.mu @ e
        grad_phi_hat = grad_phi_hat0 + (self.As @ e + grad_zt + hess_phi @ (za_dot0 + zb_dot) + h)*self.ts

        # L1AO
        sigma_hat = np.linalg.solve(hess_phi, h)
        za_dot = self.lpf(za_dot0, -sigma_hat)
        # za_dot = np.zeros(Nz)  # debug

        # Solution
        z = z0 + (za_dot + zb_dot)*self.ts
        return za_dot, grad_phi_hat, z

