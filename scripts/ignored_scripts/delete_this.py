import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

n = 3
evals = [1,2,3]
Lambda = np.diag(evals)
xi = lambda t: [1,2,t,t**2,np.sin(t)]
hat = lambda w: np.array([
    [0, -w[2], w[1]],
    [w[2], 0, -w[0]],
    [-w[1], w[0], 0]
])
Q = lambda t: expm(hat(xi(t)))
A = lambda t: Q(t)@Lambda@Q(t).T
eps = 1e-6
skew = lambda M: (M - M.T)/2
A_dot = lambda t: (A(t + eps) - A(t))/eps 
Omega = lambda t: skew(Q(t).T@(Q(t + eps) - Q(t))/eps)

t = 5
print(A_dot(t))
print((Omega(t)@Lambda + Lambda@Omega(t).T))