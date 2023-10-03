import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct
import methods
import itertools

n = 2
m = 1
A, B = methods.StabilizingGainManifold.rand_controllable_matrix_pair(n,m)
Sigma = np.eye(n)
Q = .1*np.eye(n)
R = .1*np.eye(m)
M = methods.StabilizingGainManifold(A, B, Q, R, Sigma)


K_opt = M.dlqr()
V = M.randvec(K_opt)
error = []
tspan = []
N = 1000
for t in np.linspace(-5, 5, N):
    print(t)
    K = K_opt + t*V
    
    if M.spectral_radius(M.A_cl(K)) < 1:
        H_int = np.zeros((n*m, n*m))
        H_ext = np.zeros((n*m, n*m))
        for i1, i2, j1, j2 in itertools.product(range(m), range(n), repeat=2):
            E_i = M.E(i1, i2)
            E_j = M.E(j1, j2)
            H_int_ij = M.hess_f(K, E_i, E_j)
            H_ext_ij = M.hess_f(K, E_i, E_j, Euclidean=True)
            H_int[i1*n + i2, j1*n + j2] = H_int_ij
            H_ext[i1*n + i2, j1*n + j2] = H_ext_ij
        error.append(np.linalg.norm(H_int - H_ext)**2)
        tspan.append(t)

plt.figure()
plt.semilogy(tspan, error)
plt.grid()
plt.title("Comparison of Riem. and Eucl. hessians along c(t) = K_opt + t*V")
plt.ylabel("|Hess_f(K) - Euclidean_Hess_f(K)|^2")
plt.xlabel("distance between K and K_opt")
plt.tight_layout()
plt.show()





"""Playing around with hessian of f at K_opt
K = M.dlqr()
E = M.randvec(K)
print(M.hess_f(K, E, E))
print(M.inner(K, R@E + B.T@M.P(K)@B@E, E))
"""



"""Compute hessian at each point, plot if its pos def or not
K_opt = M.dlqr()
a = np.min(K_opt)
b = np.max(K_opt)
K1_pos = []
K1_notpos = []
K2_pos = []
K2_notpos = []
for x in np.linspace(-a - 5, b + 5, 200):
    for y in np.linspace(-a - 5, b + 5, 200):
        K = np.array([[x,y]])
        if M.spectral_radius(A - B@K) < 1:
            H = []
            for i1, i2 in itertools.product(range(m), range(n)):
                E_i = M.E(i1, i2)
                H_i = M.hess_f(K,E_i, E_i)
                H.append(H_i)
                if H_i <= 0:
                    K1_notpos.append(K)
                    break

                K1_pos.append(K)

            H = []
            for i1, i2 in itertools.product(range(m), range(n)):
                E_i = M.E(i1, i2)
                H_i = M.hess_f(K,E_i, E_i, Euclidean=True)
                H.append(H_i)
                if H_i <= 0:
                    K2_notpos.append(K)
                    break
                K2_pos.append(K)

            print(np.round([x,y],2))

K1_pos = np.squeeze(np.array(K1_pos))
K1_notpos = np.squeeze(np.array(K1_notpos))
K2_pos = np.squeeze(np.array(K2_pos))
K2_notpos = np.squeeze(np.array(K2_notpos))

plt.figure()

plt.subplot(2,1,1)
plt.title("f_1")
plt.scatter(K1_pos[:,0], K1_pos[:,1], label='pos def')
if len(K1_notpos) > 1:
    plt.scatter(K1_notpos[:,0], K1_notpos[:,1], label='not pos def')
plt.scatter(K_opt[0][0], K_opt[0][1], label='K_opt', c='green')
plt.legend()
plt.grid()


plt.subplot(2,1,2)
plt.title("f_2")
plt.scatter(K2_pos[:,0], K2_pos[:,1], label='pos def')
if len(K2_notpos) > 1:
    plt.scatter(K2_notpos[:,0], K2_notpos[:,1], label='not pos def')
plt.scatter(K_opt[0][0], K_opt[0][1], label='K_opt', c='green')
plt.legend()
plt.grid()

plt.show()
"""