import numpy as np
import math

def RelMotion(X0,u,nT):
    # A是一个6x6的全0矩阵,B 是一个6x1矩阵，前三行为0，后三行为1
    A = np.zeros((6,6))
    B = np.zeros((6,3))
    A[0:3,3:6] = np.eye(3)
    A[3,0] = 3*nT**2
    A[5,2] = nT
    A[3,4] = -2*nT
    A[4,3] = -2*nT
    # 3行一列的列向量
    B[3:,:3] = np.eye(3)

    dX = A*X0 + B*u
    return dX
