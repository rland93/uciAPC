import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
# predict horizon
T = 100

# initial state
init_state = 250

# Van Heusden's Control Relevant Model
p1 = 0.98
p2 = 0.965
g = -90 * (1-p1)*(1-p2)*(1-p2)
u_TDI = 60

# A matrix used to calculate state from previous states
A = np.array([   
        [p1+2*p2, -2*p1*p2-p2*p2, p1*p2*p2],
        [1,0,0],
        [0,1,0]])

# B matrix used to calculate effect of control inputs on state
B = 1800 * g / u_TDI * np.array([[1],[0],[0]])

# "known" state is 3 vals behind current state
C = np.array([0,0,1])


''' define vars and solve optimization problem '''

# state var
x = cp.Variable((3, T+1))
u = cp.Variable((1, T))

cost = 0
constraints = []

# build costs, constraints across horizon
for t in range(T):
    cost += cp.sum_squares(x[:,t] - 100) + cp.sum_squares(u[:,t])
    constraints += [
        x[:,t+1] == A @ x[:,t] + B @ u[:,t],
        cp.norm_inf(u[:,t]) <= 1
    ]

constraints += [
    x[:,0] == init_state
]
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve(verbose=True)

fig = plt.figure()
ax = fig.add_subplot(211)
# Plot solutions

plt.plot(x[0,:].value, label='state')

plt.subplot(2,1,2)
plt.plot(u[0,:].value, label='control')

plt.show()