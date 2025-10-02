from abc import ABC, abstractmethod
import numpy as np


class DynamicalSystem(ABC):
    @abstractmethod
    def stateConstraint(self):
        pass
    
    @abstractmethod
    def getOutput(self):
        pass
    
    @abstractmethod
    def dx(self):
        pass

    @abstractmethod
    def linearize(self):
        pass


class Nth_Integrator(DynamicalSystem):
    """
    Basic N-th order integrator:
        x1dot=x2, x2dot=x3, ..., xNdot=u
        y=x1
    """
    def __init__(self, n):
        if n < 2:
            raise ValueError("Order n must be at least 2.")
        self.Nx = n
        self.Nu = 1
        self.Ny = 1

        self.A = np.eye(self.Nx, k=1)
        self.B = np.zeros((self.Nx, 1))
        self.B[-1,0] = 1.0
        self.G = np.eye(self.Nx)
        self.C = np.zeros((1, self.Nx))
        self.C[0,0] = 1.0

    def stateConstraint(self, x):
        return None
    
    def getOutput(self, x, v=None):
        if v is None:
            v = np.zeros(self.Ny)
        return self.C@x + v
    
    def dx(self, x, u, w=None):
        if w is None:
            w = np.zeros(self.Nx)
        return self.A@x + self.B@u + self.G@w

    def linearize(self, xs, us):
        return self.A, self.B, self.G, self.C

