import numpy as np
import math
from cvxopt import matrix, solvers

__author__ = 'Yankai Zhang'
# 状态量结构体(x,y,theta)·x和y单位为m·theta为车辆航向角·单位为弧度
class X:
       def __init__(self,x,y,theta):
              self.x=x
              self.y=y
              self.theta=theta

# 控制量结构体(v,delta)·delta为车辆转角·单位为弧度
class U:
       def __init__(self,v,delta):
              self.v=v
              self.delta=delta

# 矩阵求幂
def Matrixpower(A, n):
    result = A
    (a,b) =A.shape
    if n == 0:
           return np.identity(a)
    if n == 1:
        return result
    for i in range(1, n, 1):
        result = np.dot(result, A)
    return result

"""
@brief 模型预测控制
@param X 当前状态量
@param XF 当前时刻的参考状态量
@param Uf 当前时刻的参考控制量
@param Uk_1 k-1时刻的控制量（约束要用到）
@param Uf_nc 控制时域的参考控制量Ufk到Ufk+np-1的列表
@param T 状态差分方程的间隔时长
@param Np 预测时域(步长)，控制时域默认Nc=Np-1
@param a_max 车辆最大加速度
@param U_max 控制量的最大值
@param L 车辆轴距
"""
def MPC(X, Xf, Uf, Uk_1, Ufk_1,Uf_nc, T,  Np, a_max, Umax, L):

       # 车辆运动学模型     X(k+1)=AX(k)+BU(k)
       A=np.array([
              [1, 0, -T*Uf.v*math.sin(Xf.theta), T*math.cos(Xf.theta), 0],
              [0, 1, T*Uf.v*math.cos(Xf.theta), T*math.sin(Xf.theta), 0],
              [0, 0, 1, T*math.tan(Uf.delta)/L, T*Uf.v/(L*math.cos(Uf.delta)**2)],
              [0, 0, 0, 1,0],
              [0, 0, 0, 0,1]
       ])
       
       B=np.array([
             [T*math.cos(Uf.delta), 0],
             [T*math.sin(Xf.theta), 0],
             [T*math.tan(Uf.delta)/L, T*Uf.v/(L*math.cos(Uf.delta)**2)],
             [1, 0],
             [0, 1]
       ])

       C=np.array([
             [1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0]
       ])

      # 模型预测     Y=Phi*X(k)+thetaV
       '''
      Y=(X(k+1),X(k+2)、、、X(k+Np))
      Phi=(CA,CA^2,、、、CA^NP)
      Theta=[CB        0        0      0   ...   0
                    CAB     AB      0      0   ...   0
                    CA^2B CAB  CB    0   ...   0
                    ...
                    CA^Np-1B CA^Np-2B ... CB]    Theta.shape=Np*Np
       '''
       # 求Phi
       Phi=np.dot(C,A)
       for i in range(Np-1):
              temp=Matrixpower(A,i+2)
              new=np.dot(C,temp)     
              Phi=np.vstack((Phi,new))
       # 求Theta
       row_first=np.dot(C,B)
       matrix_non_zero=row_first
       Theta=np.hstack((row_first,np.zeros((3,2*(Np-1)))))
       for i in range(Np-1):
              row_first=np.dot(np.dot(C,Matrixpower(A,i+1)),B)
              matrix_non_zero=np.hstack((row_first,matrix_non_zero))
              if Np-i-2 > 0:
                     row=(np.hstack((matrix_non_zero,np.zeros((3,2*(Np-i-2))))))
              else:
                     row=matrix_non_zero
              Theta =np.vstack((Theta,row))
              
       # 目标函数    J=Y^TQY+V^TRV+pe^2 
       # 权重矩阵Q
       d = 100 * (X.x - Xf.x) ** 2 + 100 * (X.y - Xf.y) ** 2+2
       dx = 0.5+100*(X.x - Xf.x)**2
       dy = 0.5+100*(X.y - X.y)**2
       # dx=1
       # dy=1
       # d=1
       Q=np.zeros((3*Np,3*Np))
       for i in range(Np):
              Q[i*3][i*3]=dx
              Q[i*3+1][i*3+1]=dy
              Q[i*3+2][i*3+2]=d
       # 权重矩阵R
       R = np.zeros((2*Np, 2*Np))
       for i in range(0, 2*Np, 1):
              R[i][i]=0.5
       # 求目标函数的H系数矩阵
       Rho=10
       temp=np.dot(np.dot(Theta.T,Q),Theta)
       temp=temp+R
       temp=np.hstack((temp,np.zeros((2*Np,1))))
       row=np.hstack((np.zeros((1,2*Np)),np.array([[Rho]])))
       H=np.vstack((temp,row))*2
       dX=np.array([
              [X.x-Xf.x],
              [X.y-Xf.y],
              [X.theta-Xf.theta],
              [Uk_1.v-Ufk_1.v],
              [Uk_1.delta-Ufk_1.delta],
       ])
       E=np.dot(Phi,dX)
       f=np.hstack((2*np.dot(np.dot(E.T,Q),Theta),np.array([[0]])))
       # 求约束
       # 由最大加速度确定最大速度变量
       G1=np.zeros((2*Np,2*Np))
       for i in range(2*Np):
              if i%2==0:
                     k=i
                     for j in range(2*Np-i):
                            G1[k][j]=1
                            k+=1
       G1=np.hstack((G1,np.zeros((2*Np, 1))))
       G2=np.hstack((np.identity(2*Np),np.zeros((2*Np,1))))
       G=np.vstack((G1,G2))

       h1=np.zeros((2*Np,1))
       h2=np.zeros((2*Np,1))
       h3=np.zeros((2*Np,1))
       h4=np.zeros((2*Np,1))
       for i in range(Np):
              h1[2*i][0]=Umax.v-(Uk_1.v-Ufk_1.v+Uf_nc[i].v)
              h1[2*i+1][0]=Umax.delta-(Uk_1.delta-Ufk_1.delta+Uf_nc[i].delta)
              h2[2*i][0]=Umax.v+(Uk_1.v-Ufk_1.v+Uf_nc[i].v)
              h2[2*i+1][0]=Umax.delta+(Uk_1.delta-Ufk_1.delta+Uf_nc[i].delta)
       h3[0][0]=Uf_nc[0].v-Ufk_1.v
       h3[1][0]=Uf_nc[0].delta-Ufk_1.delta
       h4[0][0]=Uf_nc[0].v-Ufk_1.v
       h4[1][0]=Uf_nc[0].delta-Ufk_1.delta
       for i in range(1,Np):
              h3[i*2][0]=a_max.v*T-(Uf_nc[i].v-Uf_nc[i-1].v)
              h3[i*2+1][0]=a_max.delta*T-(Uf_nc[i].delta-Uf_nc[i-1].delta)
              h4[i*2][0]=a_max.v*T+(Uf_nc[i].v-Uf_nc[i-1].v)
              h4[i*2+1][0]=a_max.delta*T+(Uf_nc[i].delta-Uf_nc[i-1].delta)
       h1=np.vstack((h1,h3))
       h2=np.vstack((h2,h4))
       G=np.vstack((G,-G))
       h=np.vstack((h1,h2))
       H=matrix(H)
       f=matrix(f)
       G=matrix(G)
       h=matrix(h)
       # 优化求解
       sol=solvers.qp(H,f.T,G,h)
       v=sol['x'][0]+Uf.v+Uk_1.v-Ufk_1.v
       delta=sol['x'][1]+Uf.delta+Uk_1.delta-Ufk_1.delta
       return v,delta


if __name__=='__main__':
       X1=X(0,0,0)
       Xf=X(0,1,0)
       Uf=U(2,0)
       Uk_1=U(0,0)
       Ufk_1=U(0,0)
       Uf_nc=[U(2,0),U(2,0),U(2,0),U(2,0)]
       T=0.02
       Np=4
       a_max=U(5,0.041)
       Umax=U(7,20)
       L=3
       v,delta=MPC(X1, Xf, Uf, Uk_1, Ufk_1,Uf_nc, T,  Np, a_max, Umax, L)
       print(v,delta)