import numpy as np
from math import sin, cos, tan, sqrt, atan2


class LQR():

    def __init__(self, type, n, m):
        if type not in ['continuous', 'discrete']:
            raise Exception("Invalid type. Must be 'continuous' or 'discrete'.")
        self.type = type
        self.n = n
        self.m = m
        self.K = np.zeros((m, n))
    
    def setWeight(self, Q, R):
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)

    def setModel(self, A, B):
        self.A = A
        self.B = B
    
    def discretizeModel(self, h):
        """
            Simple discretization with higher-order terms ignored.
            h: sampling period (seconds)
        """
        self.A = np.eye(self.n) + self.A*h
        self.B = self.B*h

    def calculateGain(self):
        """
            Solve for P, then K.
            Continuous: solve the eigenvalue problem of the Hamiltonian.
            Discrete: use the discrete-time RDE.
        """
        if self.type == 'continuous':
            H = np.vstack((np.hstack((self.A, -self.B @ self.R_inv @ self.B.T)),
                            np.hstack((-self.Q, -self.A.T))))
            eigVal, eigVec = np.linalg.eig(H)
            U = eigVec[:, np.argsort(eigVal)]
            U = U[:, :self.n]       # 'stable' subspace - n smallest eigenvalues
            U11 = U[:self.n, :]
            U21 = U[self.n:, :]
            try:
                P = U21 @ np.linalg.inv(U11)
            except:
                # raise Exception('U11 is not invertible!')
                P = U21 @ np.linalg.pinv(U11)
            self.K = self.R_inv @ self.B.T @ P.real
    
        elif self.type == 'discrete':
            # Get K at step i: K[i] (mxn matrix)
            N_steps = 1000
            K = np.zeros((N_steps, self.m, self.n))
            S = self.Q
            for i in range(N_steps-2, -1, -1):
                R = self.R + self.B.T @ S @ self.B
                R_inv = np.linalg.inv(R)
                K[i] = R_inv @ self.B.T @ S @ self.A
                M = S - S @ self.B @ R_inv @ self.B.T @ S
                S = self.A.T @ M @ self.A + self.Q

                # check if should stop
                dK = np.divide(K[i]-K[i+1], K[i])
                max_diff = abs(np.nanmax(dK))
                if max_diff < .001:
                    # print('dLQR reached final gain at i=' + str(i))
                    break
            self.K = K[i]

    def getGain(self):
        return self.K