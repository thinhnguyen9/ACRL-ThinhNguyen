import numpy as np
from math import sin, cos, tan, sqrt, atan2
import matplotlib.pyplot as plt


class Reactor1():

    """
        Haseltine and Rawlings: “A Critical Evaluation of EKF and MHE”, 2003
        Example 1
    """

    def __init__(self):
        self.Nx = 2     # (pA, pB)
        self.Nu = 0
        self.Ny = 1     # pA + pB
        self.G = np.eye(self.Nx)
        self.C = np.array([[1., 1.]])
        self.k = 0.16

    def stateConstraint(self, x):
        return x >= np.zeros(self.Nx)
        # return None
    
    def getOutput(self, x, v=None):
        if v is None:
            v = np.zeros(self.Ny)
        return self.C @ x + v
    
    def dx(self, x, u, w=None):
        dx = np.zeros(self.Nx)
        dx[0] = -2*self.k*(x[0]**2)
        dx[1] = self.k*(x[0]**2)
        if w is None:
            w = np.zeros(self.Nx)
        return dx + w

    def linearize(self, xs, us):
        matA = np.zeros((self.Nx, self.Nx))
        matB = np.zeros((self.Nx, self.Nu))
        matA[0,0] = -4*self.k*xs[0]
        matA[1,0] = 2*self.k*xs[0]
        return matA, matB, self.G, self.C
    
