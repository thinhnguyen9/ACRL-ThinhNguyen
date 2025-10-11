# PCIP-based Moving Horizon Estimation (MHE) in Python
# -----------------------------------------------------
# - Linear time-varying plant:     x_{k+1} = A_k x_k + B_k u_k + w_k
# - Measurements (possibly LTV):   y_k     = C_k x_k + v_k
# - MHE window length: N
# - Solve at each time k a convex QP with inequality constraints using a
#   discrete-time Prediction–Correction Interior-Point (PCIP) scheme.
#
# References:
#   Fazlyab, Paternain, Preciado, Ribeiro (2018): "Prediction-Correction
#   Interior-Point Method for Time-Varying Convex Optimization"
#   (we follow their log-barrier + prediction-correction ODE, discretized).  # :contentReference[oaicite:1]{index=1}

import numpy as np
from scipy.linalg import cholesky, solve_triangular

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def build_mhe_qp(A_seq, B_seq, G_seq, C_seq, Q_inv, R_inv, x_prior, P_prior_inv,
                 u_seq, y_seq):
    """
    Optimization variable: z = [x(0),..., x(N), w(0),..., w(N-1)]

    Cost:
        J = 0.5*(x_0 - x_prior)' P_prior^{-1} (x_0 - x_prior)
            + 0.5*sum_{i=0}^{N-1} w_i' Q^{-1} w_i
            + 0.5*sum_{i=0}^{N}   (y_i - C_i x_i)' R^{-1} (y_i - C_i x_i)
    Dynamics constraints: A_eq z = b_eq
    J = 0.5 z'H z + f'z + v'(A_eq z - b_eq)  (Lagrange multipliers v)
    """
    N = len(A_seq)          # window steps N (transitions), states are N+1
    # n = A_seq[0].shape[0]
    # m = u_seq[0].shape[0]
    # p = y_seq[0].shape[0]
    n = x_prior.shape[0]

    # Quadratic terms initialization
    z_len = (2*N+1)*n   # z = [x(0),..., x(N), w(0),..., w(N-1)]
    H = np.zeros((z_len, z_len))
    f = np.zeros((z_len,))
    A_eq = np.zeros((N*n, z_len))
    b_eq = np.zeros((N*n,))

    # Prior term on x_0
    H[0:n, 0:n] += P_prior_inv
    f[0:n]      += -P_prior_inv @ x_prior

    for i in range(N):
        Ai, Bi, Gi, Ci, ui, yi = A_seq[i], B_seq[i], G_seq[i], C_seq[i], u_seq[i], y_seq[i]
        idx_xi  = slice(i*n, (i+1)*n)
        idx_xi1 = slice((i+1)*n, (i+2)*n)
        idx_wi  = slice(n*(N+1) + i*n, n*(N+1) + (i+1)*n)

        # Process noise cost
        H[idx_wi, idx_wi] += Q_inv

        # Measurement cost
        H[idx_xi, idx_xi] += Ci.T @ R_inv @ Ci
        f[idx_xi]         += -Ci.T @ R_inv @ yi

        # Dynamics constraint: x(i+1) - A(i)x(i) - G(i)w(i) = B(i)u(i)
        A_eq[i*n:(i+1)*n, idx_xi1] = np.eye(n)
        A_eq[i*n:(i+1)*n, idx_xi] = -Ai
        A_eq[i*n:(i+1)*n, idx_wi] = -Gi
        b_eq[i*n:(i+1)*n] = Bi @ ui

    idx_xN = slice(N*n, (N+1)*n)
    C = C_seq[N]
    H[idx_xN, idx_xN] += C.T @ R_inv @ C
    f[idx_xN] += -C.T @ R_inv @ y_seq[N]

    return H, f, A_eq, b_eq


def build_mhe_qp_with_lagrangian(H, f, A_eq, b_eq):

    # With Lagrange multiplier:
    # V = 0.5 z'Hz + f'z + v'(A_eq z - b_eq)
    #   = 0.5 [z;v]' [H, A_eq'; A_eq, 0] [z;v] + [f; -b_eq]' [z;v]
    v_len = A_eq.shape[0]
    H = np.block([[H, A_eq.T],
                  [A_eq, np.zeros((v_len, v_len))]])
    f = np.hstack([f, -b_eq])

    # Optional simple box constraints on states to demonstrate inequalities:
    # e.g., |x_j| <= xmax for all stacked states
    # xmax = 1e3
    # G = np.vstack([ np.eye((N+1)*n), -np.eye((N+1)*n) ])
    # h = np.hstack([ xmax*np.ones((N+1)*n), xmax*np.ones((N+1)*n) ])
    # G, h = None, None

    return H, f

# -----------------------------------------------------
# PCIP (discrete-time Euler) for inequality-constrained QP
# -----------------------------------------------------
class PCIPQP:
    """
    Prediction–Correction Interior-Point solver (discretized) for:
        minimize 0.5 z^T H z + f^T z  subject to G z < h
    Uses log-barrier with time-varying barrier c_k and prediction-correction
    as in the PCIP framework (we discretize the continuous ODE).
    """
    def __init__(self, alpha=1.0, c0=10.0, gamma_c=0.01, ts=1.0):
        self.alpha   = alpha       # correction gain (maps to P ≽ αI)
        self.c       = c0          # barrier parameter c(t) (grows with k)
        self.gamma_c = gamma_c     # c_{k+1} = c_k * exp(gamma_c)
        self.tau     = ts          # Euler step

    # ========================= with log barrier =========================
    # def _phi(self, z, H, f, G, h):
    #     # Φ(z,c) = 0.5 z'H z + f'z - (1/c) * sum log(h - Gz)
    #     s = h - G @ z
    #     if np.any(s <= 0):
    #         return np.inf
    #     return 0.5*z @ (H @ z) + f @ z - (1.0/self.c)*np.sum(np.log(s))

    # def _grad_phi(self, z, H, f, G, h):
    #     s = h - G @ z
    #     inv_s = 1.0 / s
    #     grad_barrier = (G.T @ inv_s) / self.c   # gradient of - (1/c) sum log(s)
    #     return H @ z + f + grad_barrier * (-1)  # because d/dz log(h-Gz) = -(G^T / s)

    # def _hess_phi(self, z, H, G, h):
    #     s = h - G @ z
    #     inv_s2 = 1.0 / (s*s)
    #     # Hessian of barrier term: (G^T * Diag(1/s^2) * G) / c
    #     Hb = (G.T * inv_s2) @ G / self.c
    #     return H + Hb

    # def step(self, z, H, f, G, h, dH=None, df=None, dG=None, dh=None):
    #     """
    #     One PCIP step:
    #       z_{+} = z - H^{-1}_Φ [ alpha ∇Φ + (prediction term) ]
    #     Discrete-time approximation to the continuous ODE in the paper.  # :contentReference[oaicite:2]{index=2}
    #     The prediction term uses estimated time-derivatives (finite-difference)
    #     of problem data (dH, df, dG, dh). If unknown, pass None and we omit it.
    #     """
    #     grad = self._grad_phi(z, H, f, G, h)
    #     Hphi = self._hess_phi(z, H, G, h)

    #     # Prediction term: ∂Φ/∂t  ≈ 0.5 z^T (dH) z + (df)^T z - (∂/∂t barrier)
    #     # For implementation, we need ∇_z [∂Φ/∂t]. A practical, cheap surrogate:
    #     #   g_pred ≈ (dH) z + df - (1/c) * ( -G^T ( (d(h-Gz))/s ) )
    #     # We use a simple finite-difference form if provided; otherwise set to 0.
    #     g_pred = np.zeros_like(z)
    #     if (dH is not None) or (df is not None) or (dG is not None) or (dh is not None):
    #         if dH is not None: g_pred += dH @ z
    #         if df is not None: g_pred += df
    #         if (dG is not None) or (dh is not None):
    #             s = h - G @ z
    #             ds = (0 if dh is None else dh) - (0 if dG is None else dG) @ z
    #             inv_s = 1.0 / s
    #             # gradient piece from barrier time variation
    #             g_pred += (1.0/self.c) * ( (G.T @ (ds * inv_s)) )

    #     rhs = self.alpha * grad + g_pred

    #     # Solve Hphi * delta = rhs
    #     # Use symmetric positive definite solve; add tiny reg if needed.
    #     reg = 1e-10
    #     try:
    #         delta = np.linalg.solve(Hphi + reg*np.eye(Hphi.shape[0]), rhs)
    #     except np.linalg.LinAlgError:
    #         # Fallback to least-squares
    #         delta, *_ = np.linalg.lstsq(Hphi + reg*np.eye(Hphi.shape[0]), rhs, rcond=None)

    #     z_new = z - self.tau * delta

    #     # Keep strictly feasible by backtracking if necessary
    #     s_new = h - G @ z_new
    #     bt = 0
    #     while np.any(s_new <= 0) and bt < 20:
    #         z_new = 0.5*(z + z_new)  # halve step
    #         s_new = h - G @ z_new
    #         bt += 1

    #     # Grow barrier (like c(t)=c0*e^{γ t}) per PCIP guidance  # :contentReference[oaicite:3]{index=3}
    #     self.c *= np.exp(self.gamma_c)
    #     return z_new

    # def solve(self, z0, H, f, G, h, max_iters=50):
    #     z = z0.copy()
    #     for _ in range(max_iters):
    #         z_next = self.step(z, H, f, G, h)
    #         if np.linalg.norm(z_next - z) <= self.eps*(1.0 + np.linalg.norm(z)):
    #             z = z_next
    #             break
    #         z = z_next
    #     return z
    # ========================= with log barrier =========================

    def step(self, z, H, f, dH=None, df=None):
        """
        One PCIP step:
          z+ = z - τ H^{-1} ( α∇Φ + g_pred )
        Here Φ(z) = 0.5 z'H z + f'z.
        """
        grad = H @ z + f  # ∇Φ

        g_pred = np.zeros_like(z)
        if dH is not None:
            g_pred += dH @ z
        if df is not None:
            g_pred += df

        rhs = self.alpha * grad + g_pred

        # Solve Newton system
        delta = np.linalg.solve(H, rhs)
        # delta = np.linalg.inv(H) @ rhs    # slow, not accurate

        # Euler integration step
        z_new = z - self.tau * delta
        return z_new

    def solve(self, z0, H, f, dH=None, df=None):
        grad_phi = H @ z0 + f
        hess_phi = H

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
        # if np.linalg.norm(z_next - z) <= self.eps*(1+np.linalg.norm(z)):
        #     z = z_next
        #     break
        # z = z_next
        return z

# -----------------------------------------------------
# Demo / Usage
# -----------------------------------------------------
if __name__ == "__main__":
    # Dimensions
    n, m, p = 4, 2, 3
    N = 10  # horizon

    rng = np.random.default_rng(0)

    # Time-varying model over the horizon (example)
    A_seq = [np.eye(n) + 0.01*rng.standard_normal((n,n)) for _ in range(N)]
    B_seq = [0.1*rng.standard_normal((n,m))            for _ in range(N)]
    C_seq = [rng.standard_normal((p,n))                for _ in range(N+1)]

    # Inputs and measurements for the horizon
    u_seq = [0.1*rng.standard_normal((m,)) for _ in range(N)]
    x_true = [np.zeros(n)]
    for k in range(N):
        x_true.append(A_seq[k] @ x_true[-1] + B_seq[k] @ u_seq[k])
    R_true = 0.05*np.eye(p)
    y_seq = [C_seq[k] @ x_true[k] + rng.multivariate_normal(np.zeros(p), R_true)
             for k in range(N+1)]

    # MHE weights (precision matrices)
    Q = 0.1*np.eye(n)
    R = 0.05*np.eye(p)
    Q_inv = np.linalg.inv(Q)
    R_inv = np.linalg.inv(R)

    # Prior
    x_prior = np.zeros(n)
    P_prior = 0.5*np.eye(n)
    P_prior_inv = np.linalg.inv(P_prior)

    # Build QP
    H, f, G, h = build_mhe_qp(A_seq, B_seq, C_seq, Q_inv, R_inv,
                              x_prior, P_prior_inv, u_seq, y_seq)

    # Initialize z0 (stacked states): warm start with zeros
    z0 = np.zeros(((N+1)*n,))

    # PCIP solver
    pcip = PCIPQP(alpha=1.0, c0=10.0, gamma_c=0.1, step=0.7, eps=1e-8)

    # Solve once (single window). In practice, slide the window each new sample:
    z_hat = pcip.solve(z0, H, f, G, h, max_iters=100)

    # Extract current state estimate (last state in the window)
    xN_hat = z_hat[-n:]
    print("Estimated x_N:", xN_hat)

    # --- Sliding window usage (outline) ---
    # For streaming data:
    # 1) shift window (drop oldest, append newest A,B,C,u,y),
    # 2) rebuild (or update) H,f,G,h,
    # 3) call pcip.step(...) a few times with finite-diff (dH,df,dG,dh) ≈ changes
    #    to leverage the prediction term between consecutive windows.
