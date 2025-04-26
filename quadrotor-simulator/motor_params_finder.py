import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(0, 1, 101)
kf = np.array([[0.0], [1.6], [4.0]])
kt = np.array([[0.0], [0.06], [0.12]])

uvec = np.array([[1, ui, ui**2] for ui in u])
f = uvec.dot(kf)
tau = uvec.dot(kt)

Kdt = np.polyfit(f.flatten(), tau.flatten(), 4).reshape(-1,1)
fvec = np.array([[fi[0]**4, fi[0]**3, fi[0]**2, fi[0], 1] for fi in f])
tau2 = fvec.dot(Kdt)



plt.subplot(1, 2, 1)
plt.plot(u, f, label="f")
plt.plot(u, tau, label="tau")
plt.grid(True)
plt.legend()
plt.xlabel("Motor input")
plt.ylabel("Output force")

plt.subplot(1, 2, 2)
plt.plot(f, tau, label="data")
# plt.plot(u, tau/f)
plt.plot(f, tau2, label="fit")
plt.grid(True)
plt.legend()
plt.xlabel("Output force")
plt.ylabel("Output drag torque")

plt.show()