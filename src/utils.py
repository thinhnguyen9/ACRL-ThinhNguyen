import numpy as np
from scipy.linalg import block_diag

def build_mhe_qp_with_dyn_constraints(A_seq, B_seq, G_seq, C_seq, Q_inv, R_inv, x_prior, P_prior_inv, u_seq, y_seq):
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


def build_mhe_qp_with_dyn_constraints_lagrangian(H, f, A_eq, b_eq):

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

def build_mhe_qp(A_seq, B_seq, G_seq, C_seq, Q_inv, R_inv, x_prior, P_prior_inv, u_seq, y_seq):
    """
    Linear QP with dynamics incorporated into the objective (no constraints).
    Optimization variable: z = [x(0), w(0), ..., w(N-1)]
    Objective: V = 0.5 z'Hz + f'z

    Arguments:
        A_seq = [A(0), ..., A(N-1)]
        B_seq = [B(0), ..., B(N-1)]
        G_seq = [G(0), ..., G(N-1)]
        C_seq = [C(0), ..., C(N)]
        u_seq = [u(0), ..., u(N-1)]
        y_seq = [y(0), ..., y(N)]
    """
    N = len(A_seq)
    nx = x_prior.shape[0]
    ny = y_seq[0].shape[0]
    if N>0: nu = u_seq[0].shape[0]

    A1 = np.eye((N+1)*nx)
    for i in range(N):
        A1[(i+1)*nx:(i+2)*nx, 0:(i+1)*nx] = A_seq[i] @ A1[i*nx:(i+1)*nx, 0:(i+1)*nx]
    
    G = block_diag(np.eye(nx), *G_seq)
    B = block_diag(*B_seq)
    matA = A1 @ G
    if N>0:
        matb = A1[:, nx:] @ B @ u_seq.flatten()
    else:
        matb = np.zeros((nx, ))
    matC = block_diag(*C_seq)
    matR = block_diag(*[R_inv]*(N+1))
    matQ = block_diag(np.zeros((nx,nx)), *[Q_inv]*N)

    # QP matrices
    AT_CT_R = matA.T @ matC.T @ matR
    H = matQ + AT_CT_R @ matC @ matA
    f = AT_CT_R @ (matC @ matb - y_seq.flatten())
    H[0:nx, 0:nx] += P_prior_inv
    f[0:nx]       += -P_prior_inv @ x_prior

    return H, f

