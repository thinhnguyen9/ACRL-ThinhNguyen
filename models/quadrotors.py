import numpy as np
from math import sin, cos, tan, sqrt, atan2
import matplotlib.pyplot as plt
from .basic import DynamicalSystem

class Quadrotor1(DynamicalSystem):

    """
        X-config quadrotor
        motor 1+3 clockwise, 2+4 counterclockwise
        positive z-axis downwards, positive rotations determined by right hand
    """

    def __init__(self, l, m, Jx, Jy, Jz, kf, Kdt, tc, g=9.81, Kdx=0.0, Kdy=0.0, Kdz=0.0):
        """
        Initialize the quadrotor model with physical parameters
        
        Parameters:
        -----------
        l : float
            Arm length (m)
        m : float
            Mass (kg)
        Jx, Jy, Jz : float
            Moments of inertia (kg*m^2)
        kf : array-like
            Motor input to thrust coefficients [3 elements]
            f = kf[0] + kf[1]*u + kf[2]*u^2
        Kdt : array-like
            Thrust to drag torque coefficients [5 elements]
            tau = K[0] + K[1]*f + K[2]*f^2 + K[3]*f^3 + K[4]*f^4
        tc : float
            Thrust time constant
        g : float
            Gravity (m/s^2)
        Kdx, Kdy, Kdz : float
            Drag coefficients
        """
        
        # Model dimensions
        self.Nx = 16
        self.Nu = 4
        self.Ny = 9  # can measure x, xd, y, yd, z, zd, roll, pitch, yaw
        
        # dx = Ax + Bu + Gw
        # y = Cx + v
        self.G = np.eye(self.Nx)
        self.C = np.zeros((self.Ny, self.Nx))
        self.C[0,0] = 1.
        self.C[1,1] = 1.
        self.C[2,2] = 1.
        self.C[3,3] = 1.
        self.C[4,4] = 1.
        self.C[5,5] = 1.
        self.C[6,6] = 1.
        self.C[7,7] = 1.
        self.C[8,8] = 1.

        # Model params
        self.l = l
        self.m = m
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.kf = np.array(kf)
        self.Kdt = np.array(Kdt)
        self.tc = tc
        self.g = g
        self.Kdx, self.Kdy, self.Kdz = Kdx, Kdy, Kdz
        
        # Motor params
        self.umin, self.umax = 0.0, 1.0
        self.fmax = self.u2f(self.umax)
        self.f_h = self.m*self.g/4      # hover force per motor
        self.u_h = self.f2u(self.f_h)   # hover voltage
        if self.fmax < 1.1*self.f_h:
            raise Exception("Motors are too weak, change kf!")
        
    def stateConstraint(self, x):
        """
        x: vector of variables representing the states
        """
        return None

    def u2f(self, u):
        return self.kf[0] + self.kf[1]*u + self.kf[2]*u**2
    
    def f2u(self, f):
        a, b, c = self.kf[2], self.kf[1], self.kf[0]-f
        d = sqrt(b**2 - 4*a*c)
        u = [(-b+d)/(2*a), (-b-d)/(2*a)]
        for ui in u:
            if ui >= self.umin and ui <= self.umax:
                return ui
        return 0.0

    def saturateControl(self, u):
        if len(u) != self.Nu:
            raise Exception("Invalid number of control inputs: " + str(len(u)))
        for i in range(self.Nu):
            u[i] = self.saturate(u[i], self.umin, self.umax)
        return u

    def saturate(self, val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))
    
    def gravityCompensation(self, x):
        roll, pitch = x[6], x[7]
        f = self.m * self.g / (4*cos(roll)*cos(pitch))
        return self.f2u(f)
    
    def getOutput(self, x, v=None):
        if v is None:
            v = np.zeros(self.Ny)
        return self.C @ x + v
    
    def dx(self, x, u, w=None):
        """
        Parameters:
        -----------
        x : array_like
            State vector [X, Xd, Y, Yd, Z, Zd, phi, theta, psi, p, q, r, f1, f2, f3, f4]
        w: array_like
            Process noise (disturbance): dx = Ax + Bu + Gw
        
        Returns:
        --------
        dx : numpy.ndarray
            Derivative of the state vector
        """

        # Extract state variables
        X, Xd, Y, Yd, Z, Zd = x[0], x[1], x[2], x[3], x[4], x[5]
        phi, theta, psi     = x[6], x[7], x[8]
        p, q, r             = x[9], x[10], x[11]
        f1, f2, f3, f4      = x[12], x[13], x[14], x[15]
        
        # Trigonometric functions
        s1, c1 = sin(phi), cos(phi)
        s2, c2 = sin(theta), cos(theta)
        s3, c3 = sin(psi), cos(psi)
        tg2 = np.tan(theta)
        
        # Torque calculation for yaw
        torque_psi = - (self.Kdt[4]*f1**4 + self.Kdt[3]*f1**3 + self.Kdt[2]*f1**2 + self.Kdt[1]*f1 + self.Kdt[0]) \
                     + (self.Kdt[4]*f2**4 + self.Kdt[3]*f2**3 + self.Kdt[2]*f2**2 + self.Kdt[1]*f2 + self.Kdt[0]) \
                     - (self.Kdt[4]*f3**4 + self.Kdt[3]*f3**3 + self.Kdt[2]*f3**2 + self.Kdt[1]*f3 + self.Kdt[0]) \
                     + (self.Kdt[4]*f4**4 + self.Kdt[3]*f4**3 + self.Kdt[2]*f4**2 + self.Kdt[1]*f4 + self.Kdt[0])
        
        # Initialize derivative vector
        dx = np.zeros(self.Nx)
        
        # State derivatives
        dx[0] = Xd
        dx[1] = (-(s1*s3+c1*c3*s2)*(f1+f2+f3+f4) - self.Kdx*Xd)/self.m
        dx[2] = Yd
        dx[3] = (-(c1*s2*s3-c3*s1)*(f1+f2+f3+f4) - self.Kdy*Yd)/self.m
        dx[4] = Zd
        dx[5] = self.g - (c1*c2*(f1+f2+f3+f4) - self.Kdz*Zd)/self.m
        dx[6] = p + s1*tg2*q + c1*tg2*r
        dx[7] = c1*q - s1*r
        dx[8] = s1/c2*q + c1/c2*r
        dx[9] = ((self.Jy-self.Jz)*q*r + self.l/sqrt(2)*(f1-f2-f3+f4))/self.Jx
        dx[10] = ((self.Jz-self.Jx)*p*r + self.l/sqrt(2)*(f1+f2-f3-f4))/self.Jy
        dx[11] = ((self.Jx-self.Jy)*p*q + torque_psi)/self.Jz
        dx[12] = -1/self.tc*f1 + 1/self.tc*(self.kf[2]*u[0]**2 + self.kf[1]*u[0] + self.kf[0])
        dx[13] = -1/self.tc*f2 + 1/self.tc*(self.kf[2]*u[1]**2 + self.kf[1]*u[1] + self.kf[0])
        dx[14] = -1/self.tc*f3 + 1/self.tc*(self.kf[2]*u[2]**2 + self.kf[1]*u[2] + self.kf[0])
        dx[15] = -1/self.tc*f4 + 1/self.tc*(self.kf[2]*u[3]**2 + self.kf[1]*u[3] + self.kf[0])

        # Add process noise
        if w is None:
            w = np.zeros(self.Nx)
        for i in range(self.Nx):
            dx[i] += w[i]
        
        return dx

    def linearize(self, xs, us):
        """
        Linearize quadcopter dynamics around equilibrium point (xs, us).
        Linearized model:
        dx = Ax + Bu + Gw
        y = Cx + v
        
        Parameters:
        -----------
        xs : array_like
            State vector at equilibrium point
        us : array_like
            Input vector at equilibrium point
        
        Returns:
        --------
        A : Linearized system matrix A
        B : Linearized input matrix B
        G : Process noise matrix G
        C : Output matrix C
        """
        
        # Extract state variables
        X, Xd, Y, Yd, Z, Zd = xs[0], xs[1], xs[2], xs[3], xs[4], xs[5]
        phi, theta, psi     = xs[6], xs[7], xs[8]
        p, q, r             = xs[9], xs[10], xs[11]
        f1, f2, f3, f4      = xs[12], xs[13], xs[14], xs[15]
        u1, u2, u3, u4      = us[0], us[1], us[2], us[3]
        
        # Trigonometric functions
        s1, c1 = sin(phi), cos(phi)
        s2, c2 = sin(theta), cos(theta)
        s3, c3 = sin(psi), cos(psi)
        tg2 = np.tan(theta)
        
        # Initialize matrices
        matA = np.zeros((self.Nx, self.Nx))
        matB = np.zeros((self.Nx, self.Nu))
        
        # Fill in matrix A (system matrix)
        # Row 1
        matA[0, 1] = 1
        
        # Row 2
        matA[1, 1] = -self.Kdx/self.m
        matA[1, 6] = -((c1*s3-c3*s1*s2)*(f1+f2+f3+f4))/self.m
        matA[1, 7] = -(c1*c3*c2*(f1+f2+f3+f4))/self.m
        matA[1, 8] = -((c3*s1-c1*s3*s2)*(f1+f2+f3+f4))/self.m
        matA[1, 12] = -(s1*s3+c1*c3*s2)/self.m
        matA[1, 13] = -(s1*s3+c1*c3*s2)/self.m
        matA[1, 14] = -(s1*s3+c1*c3*s2)/self.m
        matA[1, 15] = -(s1*s3+c1*c3*s2)/self.m
        
        # Row 3
        matA[2, 3] = 1
        
        # Row 4
        matA[3, 3] = -self.Kdy/self.m
        matA[3, 6] = ((c1*c3+s1*s3*s2)*(f1+f2+f3+f4))/self.m
        matA[3, 7] = -(c1*c2*s3*(f1+f2+f3+f4))/self.m
        matA[3, 8] = -((s1*s3+c1*c3*s2)*(f1+f2+f3+f4))/self.m
        matA[3, 12] = (c3*s1-c1*s3*s2)/self.m
        matA[3, 13] = (c3*s1-c1*s3*s2)/self.m
        matA[3, 14] = (c3*s1-c1*s3*s2)/self.m
        matA[3, 15] = (c3*s1-c1*s3*s2)/self.m
        
        # Row 5
        matA[4, 5] = 1
        
        # Row 6
        matA[5, 5] = self.Kdz/self.m
        matA[5, 6] = (c2*s1*(f1+f2+f3+f4))/self.m
        matA[5, 7] = (c1*s2*(f1+f2+f3+f4))/self.m
        matA[5, 12] = -(c1*c2)/self.m
        matA[5, 13] = -(c1*c2)/self.m
        matA[5, 14] = -(c1*c2)/self.m
        matA[5, 15] = -(c1*c2)/self.m
        
        # Row 7
        matA[6, 6] = q*c1*tg2 - r*s1*tg2
        matA[6, 7] = r*c1*(tg2**2+1) + q*s1*(tg2**2+1)
        matA[6, 9] = 1
        matA[6, 10] = s1*tg2
        matA[6, 11] = c1*tg2
        
        # Row 8
        matA[7, 6] = -r*c1 - q*s1
        matA[7, 10] = c1
        matA[7, 11] = -s1
        
        # Row 9
        matA[8, 6] = (q*c1)/c2 - (r*s1)/c2
        matA[8, 7] = (r*c1*s2)/c2**2 + (q*s1*s2)/c2**2
        matA[8, 10] = s1/c2
        matA[8, 11] = c1/c2
        
        # Row 10
        matA[9, 10] = (r*(self.Jy-self.Jz))/self.Jx
        matA[9, 11] = (q*(self.Jy-self.Jz))/self.Jx
        matA[9, 12] = (2**(1/2)*self.l)/(2*self.Jx)
        matA[9, 13] = -(2**(1/2)*self.l)/(2*self.Jx)
        matA[9, 14] = -(2**(1/2)*self.l)/(2*self.Jx)
        matA[9, 15] = (2**(1/2)*self.l)/(2*self.Jx)
        
        # Row 11
        matA[10, 9] = -(r*(self.Jx-self.Jz))/self.Jy
        matA[10, 11] = -(p*(self.Jx-self.Jz))/self.Jy
        matA[10, 12] = (2**(1/2)*self.l)/(2*self.Jy)
        matA[10, 13] = (2**(1/2)*self.l)/(2*self.Jy)
        matA[10, 14] = -(2**(1/2)*self.l)/(2*self.Jy)
        matA[10, 15] = -(2**(1/2)*self.l)/(2*self.Jy)
        
        # Row 12
        matA[11, 9] = (q*(self.Jx-self.Jy))/self.Jz
        matA[11, 10] = (p*(self.Jx-self.Jy))/self.Jz
        matA[11, 12] = -(4*self.Kdt[4]*f1**3+3*self.Kdt[3]*f1**2+2*self.Kdt[2]*f1+self.Kdt[1])/self.Jz
        matA[11, 13] = (4*self.Kdt[4]*f2**3+3*self.Kdt[3]*f2**2+2*self.Kdt[2]*f2+self.Kdt[1])/self.Jz
        matA[11, 14] = -(4*self.Kdt[4]*f3**3+3*self.Kdt[3]*f3**2+2*self.Kdt[2]*f3+self.Kdt[1])/self.Jz
        matA[11, 15] = (4*self.Kdt[4]*f4**3+3*self.Kdt[3]*f4**2+2*self.Kdt[2]*f4+self.Kdt[1])/self.Jz
        
        # Rows 13-16
        matA[12, 12] = -1/self.tc
        matA[13, 13] = -1/self.tc
        matA[14, 14] = -1/self.tc
        matA[15, 15] = -1/self.tc
        
        # Fill in matrix B (input matrix)
        matB[12, 0] = (self.kf[1] + 2*self.kf[2]*u1)/self.tc
        matB[13, 1] = (self.kf[1] + 2*self.kf[2]*u2)/self.tc
        matB[14, 2] = (self.kf[1] + 2*self.kf[2]*u3)/self.tc
        matB[15, 3] = (self.kf[1] + 2*self.kf[2]*u4)/self.tc
        
        return matA, matB, self.G, self.C
    
    
