import numpy as np
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load the simulation data from the .npy file
file_path = os.path.join(os.path.dirname(__file__), '66d59550.npy')
with open(file_path, 'rb') as f:
    tvec = np.load(f)
    xvec = np.load(f)
    xhat = np.load(f)
    yvec = np.load(f)
    uvec = np.load(f) 