import numpy as np
import matplotlib.pyplot as plt
import Orbit_Dynamics.CW_Prop as CW
from control_dep import discrete_state ,MPC_dep, quadprog

# ship and simulation parameters

Tf = 2000   #Total simulation time
dT = 5       #Sampling period
thrust_max = 1e-3
mu = 398600.5e9
sma = 7108e3
nT = np.sqrt(mu/sma**3)
# 动作维度
n_obs = 6 
n_state = 6
n_input = 3
# 预测 horizon
ny = 35

A,B = discrete_state(dT,nT)
C = np.eye(n_obs)
R = np.eye(3) * 1e6
Q11 = np.eye(3) * 1e1
Q22 = np.eye(3) * 2
Q = np.block([[Q11, np.zeros((3, 3))], [np.zeros((3, 3)), Q22]])
S = np.eye(n_obs)
mpc = MPC_dep(n_state, n_obs, ny, A, B, C, R, Q, S)
H = mpc.H_func()
f = mpc.f_func()
options = {'disp': False}

Xi_0 = CW.orbitgen(nT,0.1*np.pi,0.2*np.pi,200)
Xf_0 = CW.orbitgen(nT,0.4*np.pi,0.1*np.pi,200)

Xf = CW.Free(0*np.pi/nT,0,Xf_0,nT)
x = np.zeros((n_obs, int(np.ceil(Tf/dT))+1))            # State vector
delta_u = np.zeros((3, int(np.ceil(Tf/dT))))     # Input increment vector

x[:,0] = CW.Free(-0.4*np.pi/nT,0,Xi_0,nT);  

dV = 0
for t, i in zip(np.arange(0, Tf, dT) , range(0,int(np.ceil(Tf/dT)))):
    # 预测的目标状态
    r = np.zeros(ny*n_obs)
    for ri in range(1, ny+1):
        t_r = t + (ri-1)*dT
        r[ri*n_obs-n_obs:ri*n_obs] = CW.Free(t_r, 0, Xf, nT)
    
    # 推力限制
    Mbar = mpc.Mbar_func()
    Acons = mpc.Acons_func() @ mpc.Fbar_func() 
    # ww = mpc.Acons_func()
    # cc = mpc.Acons_func()@ Mbar @ x[:, i].reshape(6, 1)
    # qq = mpc.Dy_func(-np.ones((6,1))*4e3, np.ones((6,1))*4e3)
    Bcons = mpc.Dy_func(-np.ones((n_obs,1))*4e3, np.ones((n_obs,1))*4e3) - mpc.Acons_func()@ Mbar @ x[:, i].reshape(n_obs, 1)
    lb = -np.ones([n_input*ny,1])*thrust_max
    ub = np.ones([n_input*ny,1])*thrust_max
    
    # 解控制
    Hbar = mpc.H_func()
    Hbar_half = Hbar/2 + Hbar.T/2
    fo = np.hstack((x[:, i], r)).T @ f # first ordered coefficient 
    delta_uFut = quadprog(Hbar_half, fo.T, Acons, Bcons,  None, None,lb, ub)
    # delta_uFut = cvxopt.solvers.qp(Hbar_half, fo.T, L, k, Aeq, beq)
    # if exitflag == -2:
    #     break
    
    delta_u[:, i] = delta_uFut[0:n_input].flatten() # 获取要应用的输入增量
    x[:, i+1] = A @ x[:, i] + B @ delta_u[:, i]  # 计算下一个状态以更新位置系统
    dV += np.linalg.norm(delta_u[:, i])
    # print(x[:, i+1])


t_period = np.arange(0, 2*np.pi/nT, 10)
Xi_orbit = np.zeros((6, len(t_period)))
Xf_orbit = np.zeros((6, len(t_period)))
for j in range(0, len(t_period)):
    Xi_orbit[:, j] = CW.Free(t_period[j], 0, Xi_0, nT)
    Xf_orbit[:, j] = CW.Free(t_period[j], 0, Xf_0, nT)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
ax.plot(Xf_orbit[0, :], Xf_orbit[1, :], Xf_orbit[2, :], 'r')
ax.plot(x[0, :], x[1, :], x[2, :], 'b--')
ax.plot(x[0, 0], x[1, 0], x[2, 0], 'ko')
ax.plot(x[0, -1], x[1, -1], x[2, -1], 'ro')
ax.plot(Xf_orbit[0, 0], Xf_orbit[1, 0], Xf_orbit[2, 0], 'rs')
ax.legend(['InitialOrbit', 'TargetOrbit', 'Reconfig.Traj.', 'Dep.', 'Arriv.', 'Planned'])
ax.grid(True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# # 假设你的数据范围是从min_val到max_val
# min_val = min(min(Xi_orbit[:, 0]), min(Xi_orbit[:, 1]), min(Xi_orbit[:, 2]), 
#                 min(Xf_orbit[:, 0]), min(Xf_orbit[:, 1]), min(Xf_orbit[:, 2]), 
#                 min(x[:, 0]), min(x[:, 1]), min(x[:, 2]))

# max_val = max(max(Xi_orbit[:, 0]), max(Xi_orbit[:, 1]), max(Xi_orbit[:, 2]), 
#                 max(Xf_orbit[:, 0]), max(Xf_orbit[:, 1]), max(Xf_orbit[:, 2]), 
#                 max(x[:, 0]), max(x[:, 1]), max(x[:, 2]))

# ax.set_xlim([min_val, max_val])
# ax.set_ylim([min_val, max_val])
# ax.set_zlim([min_val, max_val])
plt.axis('equal')
plt.show()