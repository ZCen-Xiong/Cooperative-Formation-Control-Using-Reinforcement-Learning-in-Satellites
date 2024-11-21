import math
import numpy as np 
'''
@File         :CW_Prop.py
@Description  : CW_Prop：CW方程预报
@Time         :2024/01/28 21:32:01
@Author       :ZicenXiong
@Version      :1.0
'''
def orbitgen(nT, azi_0, ele_0, ref_p):
    # FIRST LVLH 
    '''
    @File         :from elevation and azimuth to generate FIRST LVLH orbit
    @Description  : CW_Prop：CW方程预报
    @Time         :2024/01/28 21:32:01
    @Author       :ZicenXiong
    @Version      :1.0
    '''
    ele = ele_0
    azi = azi_0

    if ele_0 < 0:
        ele = -ele_0
        azi = azi_0 - np.pi
        while azi < 0:
            azi += 2*np.pi

    z_l_cclw = azi - np.pi
    z_l = -z_l_cclw
    z_0 = np.pi/2 + z_l

    p = ref_p
    X0 = np.array([-p, 0, 0, 0, 2*p*nT, 0])

    z_cw = np.arctan2(np.sin(z_0)/2, np.cos(z_0))
    X_p_temp = Free(z_cw/nT, 0, X0, nT)

    X_h_temp = Free((np.pi/2)/nT, 0, X_p_temp, nT)
    v1 = np.array([np.cos(azi) * np.cos(ele),np.sin(azi) * np.cos(ele),np.sin(ele)]) * 200
    s = (-v1[0]*X_h_temp[0] - v1[1]*X_h_temp[1]) / v1[2]

    while np.sqrt(0.5*(3*p**2 + s**2 + np.sqrt(9*p**4 + 6*p**2*s**2*np.cos(2*z_cw) + s**4)) + p**2) > ref_p*1.5:
        p = 0.95*p
        s = 0.95*s

    X0 = np.array([-p, 0, 0, 0, 2*p*nT, 0])
    z_cw = np.arctan2(np.sin(z_0)/2, np.cos(z_0))
    X_p = Free(z_cw/nT, 0, X0, nT)
    X_p[5] = s*nT

    return X_p

def Free(t, t0, X0, nT):
    
    tau = nT * (t - t0)
    # first LVLH
    F = np.array([[4 - 3*np.cos(tau), 0, 0, np.sin(tau)/nT, -(2*np.cos(tau) - 2)/nT, 0],
                [6*np.sin(tau) - 6*tau, 1, 0, (2*np.cos(tau) - 2)/nT, (4*np.sin(tau) - 3*tau)/nT, 0],
                [0, 0, np.cos(tau), 0, 0, np.sin(tau)/nT],
                [3*nT*np.sin(tau), 0, 0, np.cos(tau), 2*np.sin(tau), 0],
                [6*nT*(np.cos(tau) - 1), 0, 0, -2*np.sin(tau), 4*np.cos(tau) - 3, 0],
                [0, 0, -nT*np.sin(tau), 0, 0, np.cos(tau)]])
    
    X = F@X0

    return X

def WithImpulse(t, tp, t0, X0, DeltaV, nT, N):
    """
    @Description : 有脉冲的CW外推
    @Param:
    @Returns     :
    
    """
    tau = nT * (t - t0)
    # second LVLH
    F_rr = np.array([[1.0, 0.0, 6.0 * (tau - np.sin(tau))],
                        [0.0, np.cos(tau), 0.0],
                        [0.0, 0.0, 4.0 - 3.0 * np.cos(tau)]])

    F_rv = np.array([[(4.0 * np.sin(tau) - 3.0 * tau) / nT, 0.0, 2.0 * (1.0 - np.cos(tau)) / nT],
                        [0.0, np.sin(tau) / nT, 0.0],
                        [2.0 * (np.cos(tau) - 1.0) / nT, 0.0, np.sin(tau) / nT]])

    F_vr = np.array([[0.0, 0.0, 6.0 * nT * (1.0 - np.cos(tau))],
                        [0.0, -nT * np.sin(tau), 0.0],
                        [0.0, 0.0, 3.0 * nT * np.sin(tau)]])

    F_vv = np.array([[4.0 * np.cos(tau) - 3.0, 0.0, 2.0 * np.sin(tau)],
                        [0.0, np.cos(tau), 0.0],
                        [-2.0 * np.sin(tau), 0.0, np.cos(tau)]])

    F = np.zeros((6, 6))
    F[:3, :3] = F_rr
    F[:3, 3:] = F_rv
    F[3:, :3] = F_vr
    F[3:, 3:] = F_vv

    X = np.dot(F, X0)

    for i in range(N):
        tauv = nT * (t - tp[i] - t0)

        F_rv_i = np.array([[(4.0 * np.sin(tauv) - 3.0 * tauv) / nT, 0.0, 2.0 * (1.0 - np.cos(tauv)) / nT],
                            [0.0, np.sin(tauv) / nT, 0.0],
                            [2.0 * (np.cos(tauv) - 1.0) / nT, 0.0, np.sin(tauv) / nT]])

        F_vv_i = np.array([[4.0 * np.cos(tauv) - 3.0, 0.0, 2.0 * np.sin(tauv)],
                            [0.0, np.cos(tauv), 0.0],
                            [-2.0 * np.sin(tauv), 0.0, np.cos(tauv)]])

        F_V = np.hstack((F_rv_i.T, F_vv_i.T)).T

        X += np.dot(F_V, DeltaV[:, i]) * (1.0 if t >= tp[i] else 0.0)

    return X

def CW_Prop(dr0,dv0,Omega,t):
    dr0 = np.reshape(dr0,3)
    dv0 = np.reshape(dv0, 3)
    x0 = dr0[0]
    y0 = dr0[1]
    z0 = dr0[2]
    vx0 = dv0[0]
    vy0 = dv0[1]
    vz0 = dv0[2]

    x = x0+2*vz0/Omega*(1-math.cos(Omega*t))+(4*vx0/Omega-6*z0)*math.sin(Omega*t)+(6*Omega*z0-3*vx0)*t
    y = y0*math.cos(Omega*t)+vy0/Omega*math.sin(Omega*t)
    z = 4*z0-2*vx0/Omega+(2*vx0/Omega-3*z0)*math.cos(Omega*t)+vz0/Omega*math.sin(Omega*t)
    vx = (4*vx0-6*Omega*z0)*math.cos(Omega*t)+2*vz0*math.sin(Omega*t)+6*Omega*z0-3*vx0
    vy = vy0*math.cos(Omega*t)-y0*Omega*math.sin(Omega*t)
    vz = vz0*math.cos(Omega*t)+(3*Omega*z0-2*vx0)*math.sin(Omega*t)

    dr = np.array([x, y, z])
    dv = np.array([vx, vy, vz])

    return dr,dv

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    mu = 398600.5e9
    sma = 7108e3
    nT = np.sqrt(mu/sma**3)
    azi_0 = 0.4*np.pi
    ele_0 = 0.2*np.pi
    ref_p = 200

    X_p = orbitgen(nT, azi_0, ele_0, ref_p)
    print(X_p)
    t_period = np.arange(0, 2*np.pi/nT, 10)
    Xi_orbit = np.zeros((6, len(t_period)))
    Xi_orbit[:, 0] = X_p
    for j in range(1, len(t_period)):
        Xi_orbit[:, j] = Free(t_period[j], 0, X_p, nT)

    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
    ax.plot(X_p[0], X_p[1], X_p[2], 'o')
    def vec_plot(azi, ele):
        vec = vec = np.array([[0, np.cos(azi) * np.cos(ele)],
                    [0, np.sin(azi) * np.cos(ele)],
                    [0, np.sin(ele)]]) * 200
        ax.plot(vec[0, :], vec[1, :], vec[2, :], 'r')
        return vec[1,:]

    v1 = vec_plot(azi_0, ele_0)

    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()