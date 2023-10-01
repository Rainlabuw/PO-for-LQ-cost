import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct
import methods
import math
import itertools


n = 4
m = 3
A, B = methods.StabilizingGainManifold.rand_controllable_matrix_pair(n,m)
Sigma = np.eye(n)
Q = np.eye(n)
R = np.eye(m)
M = methods.StabilizingGainManifold(A, B, Q, R, Sigma)

K = M.rand()

while True:
    V = M.randvec(K)

    tspan = []
    error = []
    for t in np.linspace(-3,3,1000):
        K_t = K + t*V
        if M.spectral_radius(M.A_cl(K_t)) < 1 and M.f(K_t) < 20:
            tspan.append(t)
            error.append(math.exp(M.f(K_t)))

    dd_error = np.diff(error, 2)
    if np.min(dd_error) < 0:
        break


plt.subplot(2,1,1)
plt.plot(tspan, error)
plt.title("f(K + tV)")
plt.grid()

plt.subplot(2,1,2)
plt.plot(tspan[:-2], dd_error)
plt.grid()
plt.show()





"""Compute sectional curvature at random K along axes E_i and E_j.
K = M.dlqr()
for i1, i2, j1, j2 in itertools.product(range(m), range(n), repeat=2):
    if (i1,i2) != (j1,j2):
        sc = M.sectional_curvature(K, i1, i2, j1, j2)
        print(sc)
"""


# Spencer: compute geodesics c(t), and plot f(c(t))
# V = M.randvec(K)
# def eom(t, y):
#     K = np.reshape(y[:m*n], (m,n))
#     K_dot = np.reshape(y[m*n:], (m,n))
#     K_dot = K_dot/M.norm(K,K_dot)

#     print(
#         f"time: {t}, " +
#         f"rho(A-BK): {M.spectral_radius(A - B@K)}, " +
#         f"||K_dot||: {M.norm(K, K_dot)}"
#     )
   
#     K_ddot = M.zerovec()
#     for i1, j1, k1 in itertools.product(range(m), repeat=3):
#         for i2, j2, k2 in itertools.product(range(m), repeat=3):
#             K_ddot[k1,k2] += -M.Gamma(K, i1, i2, j1, j2, k1, k2) * \
#                 K_dot[i1,i2]*K_dot[j1,j2]
#     y_dot = np.reshape(np.concatenate([K_dot, K_ddot]), (-1,))
#     return y_dot

# y0 = np.reshape(np.concatenate([K, V]), (-1,))
# T = 1
# sol = solve_ivp(eom, [0,T], y0, method='LSODA', rtol=1e-8, atol=1e-8)
# t = sol['t']
# y = sol['y']
# N = len(t)
# K_traj = y[:m*n, :]

# out = []
# for k in range(N):
#     K = K_traj[:,k]
#     K = np.reshape(K, (m,n))
#     out.append(M.f(K))
 

# plt.plot(t, out)
# plt.grid()
# plt.show()