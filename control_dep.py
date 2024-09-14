import numpy as np
from scipy.signal import cont2discrete
import cvxopt

def discrete_state(dT, nT):
    Ac = np.zeros((6, 6))
    Bc = np.zeros((6, 3))
    
    Ac[0:3, 3:6] = np.eye(3)
    Ac[3, 0] = 3 * nT**2
    Ac[3, 4] = 2 * nT
    Ac[4, 3] = -2 * nT
    Ac[5, 2] = -nT**2
    
    Bc[3:6, 0:3] = np.eye(3)
    
    C = np.eye(6)
    D = np.zeros((6, 3))
    
    sys_c = cont2discrete((Ac, Bc, C, D), dT, method='zoh')
    
    Ad = sys_c[0]
    Bd = sys_c[1]
    
    return Ad, Bd

class MPC_dep:
    def __init__(self, n_state, n_observ, ny, A, B, C, R, Q, S):
        self.n_state = n_state
        self.n_observ = n_observ
        self.ny = ny
        self.A = A 
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.S = S

    def Qbar_func(self):
        # 计算 位置代价矩阵的增广
        ny = self.ny
        C = self.C
        S = self.S
        Q = self.Q
        Qbar_tmp = [[None for _ in range(ny)] for _ in range(ny)]
        for i in range(ny):
            for j in range(ny):
                if i == j:
                    if i == ny - 1:
                        Qbar_tmp[i][j] = np.dot(np.dot(C.T, S), C)
                    else:
                        Qbar_tmp[i][j] = np.dot(np.dot(C.T, Q), C)
                else:
                    Qbar_tmp[i][j] = np.zeros_like(np.dot(C.T, Q).dot(C))

        Qbar = np.block(Qbar_tmp)
        return Qbar
    
    def Tbar_func(self):
        # 计算 Q*C
        ny = self.ny
        S = self.S
        C = self.C
        Q = self.Q
        Tbar_tmp = np.empty((ny, ny), dtype=object)
        
        for i in range(ny):
            for j in range(ny):
                if i == j:
                    if i == ny-1:
                        Tbar_tmp[i, j] = S @ C
                    else:
                        Tbar_tmp[i, j] = Q @ C
                else:
                    Tbar_tmp[i, j] = np.zeros_like(Q @ C)
        
        Tbar = np.block([[Tbar_tmp[i, j] for j in range(ny)] for i in range(ny)])
        
        return Tbar

    def Rbar_func(self):
        # 计算控制代价矩阵
        ny = self.ny
        R = self.R
        Rbar_tmp = np.empty((ny, ny), dtype=object)
        
        for i in range(ny):
            for j in range(ny):
                if i == j:
                    Rbar_tmp[i, j] = R
                else:
                    Rbar_tmp[i, j] = np.zeros_like(R)
        
        Rbar = np.block([[Rbar_tmp[i, j] for j in range(ny)] for i in range(ny)])
        return Rbar

    def Fbar_func(self):
        # 计算状态转移矩阵 AB在整个horizon内的影响
        ny = self.ny
        A = self.A
        B = self.B
        
        Cbar_tmp = np.empty((ny, ny), dtype=object)
        tmp = B.copy()
        
        for i in range(ny):
            for k in range(i):
                Cbar_tmp[k, i] = np.zeros_like(B)
            
            for j in range(i, ny):
                Cbar_tmp[j, i] = tmp
                tmp = A @ tmp
            
            tmp = B.copy()
        
        Fbar = np.block([[Cbar_tmp[j, i] for i in range(ny)] for j in range(ny)])
        
        return Fbar
    
    def Mbar_func(self):
        ny = self.ny
        A = self.A
        Mbar_tmp = np.empty((ny, 1), dtype=object)
        tmp = A.copy()
        
        for i in range(ny):
            Mbar_tmp[i, 0] = tmp
            tmp = tmp @ A
        
        Mbar = np.block([[Mbar_tmp[i, 0]] for i in range(ny)])
        
        return Mbar

    def H_func(self):
        # 计算u的二次项
        Fbar = self.Fbar_func()
        Qbar = self.Qbar_func()
        Rbar = self.Rbar_func()
        Hbar = Fbar.T @ Qbar @ Fbar + Rbar
        return Hbar
    
    def f_func(self):
        # 计算u的一次项
        Mbar = self.Mbar_func()
        Fbar = self.Fbar_func()
        Tbar = self.Tbar_func()
        Qbar = self.Qbar_func()
        fbar = np.vstack((Mbar.T @ Qbar @ Fbar, -Tbar @ Fbar))
        return fbar

    def Acons_func(self):
        ny = self.ny
        n_state = self.n_state

        Cy_tmp1 = [[None for _ in range(ny)] for _ in range(ny)]
        Cy_tmp2 = [[None for _ in range(ny)] for _ in range(ny)]
        
        for i in range(ny):
            for j in range(ny):
                if i == j:
                    Cy_tmp1[i][j] = np.eye(n_state)
                    Cy_tmp2[i][j] = -np.eye(n_state)
                else:
                    Cy_tmp1[i][j] = np.zeros((n_state, n_state))
                    Cy_tmp2[i][j] = np.zeros((n_state, n_state))
        
        Cy = np.block([[np.block(Cy_tmp1)], [np.block(Cy_tmp2)]])
        
        return Cy


    def Dy_func(self, Xmin, Xmax):
        ny = self.ny

        Dy_tmp = [[None] for _ in range(2*ny)]
        
        for i in range(2*ny):
            if i < ny:
                Dy_tmp[i][0] = Xmax
            else:
                Dy_tmp[i][0] = -Xmin
        
        Dy = np.concatenate([np.array(Dy_tmp[i][0]) for i in range(2*ny)])
        
        return Dy


def quadprog(H, f, A=None, B=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    cvxopt.solvers.options['show_progress'] = False
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if A is not None or B is not None:
        assert(B is not None and A is not None)
        if lb is not None:
            A = np.vstack([A, -np.eye(n_var)])
            B = np.vstack([B, -lb])

        if ub is not None:
            A = np.vstack([A, np.eye(n_var)])
            B = np.vstack([B, ub])

        A = cvxopt.matrix(A, tc='d')
        B = cvxopt.matrix(B, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, A, B, Aeq, beq)

    return np.array(sol['x'])


# if __name__ == '__main__':
#     H=np.array([[1,-1],[-1,2]])
#     print(H)
#     f=np.array([[-2],[-6]])
#     print(f)
#     L=np.array([[1,1],[-1,2],[2,1]])
#     print(L)
#     k=np.array([[2],[2],[3]])
#     print(k)
#     res=quadprog(H, f, L,k)
#     print(res)



if __name__ == "__main__":
    # 调用函数并打印结果
    # dT = 0.1
    # nT = 10
    # Ad, Bd = discrete_state(dT, nT)
    # print("Ad:")
    # print(Ad)
    # print("Bd:")
    # print(Bd)

    # 测试MPC_dep类
    n_observ = 6
    n_state = 6
    ny = 20
    Tfinal = 2000   #Total simulation time
    dT = 5       #Sampling period
    thrust_max = 1e-3
    mu = 398600.5e9
    sma = 7108e3
    nT = np.sqrt(mu/sma**3)
    A,B = discrete_state(dT,nT)
    C = np.eye(6)
    R = np.eye(3) * 1e6
    Q11 = np.eye(3) * 1e1
    Q22 = np.eye(3) * 2
    Q = np.block([[Q11, np.zeros((3, 3))], [np.zeros((3, 3)), Q22]])
    S = np.eye(6)
    
    mpc = MPC_dep(n_state, n_observ, ny, A, B, C, R, Q, S)
    H = mpc.H_func()
    f = mpc.f_func()
    print("H:") 
    print(H[0:6, 0:6])
    print(H[54:60, 54])
    print("f:")
    print(f[0:3,0:3])
    print(f[117:120,0:3])
    