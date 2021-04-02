#import sys
#sys.path.append("/home/zyk/mpc/lib")
import numpy as np
import mpc
import loadwaypoints
import carmodel

# 存储上一时刻控制量
with open("/home/zyk/mpc/resources/tmpUk_1.txt", 'w') as f:
       f.write("0 0\n")
# 日志
loggotxt = open("/home/zyk/mpc/resources/loggo.txt", 'w')
loggotxt.close()
#时间线

timestemps=np.arange(0,55,1)
#路点
waypoints=loadwaypoints.GetRefTrack("/home/zyk/mpc/resources/linewaypoint/line200.xlsx")
X=mpc.X(-97.8251783,-615.887209,0.348193399)
Uk_1=mpc.U(0,0)
Ufk_1=mpc.U(0,0)
index_nowref=0
Np=20
L=2.95
for t in timestemps:
       #根据当前状态和参考路点选择当前参考点及Np个参考点waypoints_nc
       distance = 10000
       (len_waypoints,b)=waypoints.shape
       for i in range(index_nowref+1,len_waypoints):
              if (waypoints[i, 0] - X.x)**2+(waypoints[i, 1] - X.y)**2 < distance:
                     distance = (waypoints[i, 0] - X.x)**2+(waypoints[i, 1] - X.y)**2
                     temp = i

       index_nowref=temp
       waypoints_nc=[]
       Uf_nc=[]
       for i in range(Np):
              if index_nowref+i <= len_waypoints-1:
                     waypoints_nc.append(waypoints[index_nowref+i])
              else:
                     waypoints_nc.append(waypoints[len_waypoints-1])
              Uf_nc.append(mpc.U(waypoints_nc[i][3],waypoints_nc[i][4]))
       Xf=mpc.X(waypoints_nc[0][0],waypoints_nc[0][1],waypoints_nc[0][2])
       Uf=mpc.U(waypoints_nc[0][3],waypoints_nc[0][4])
       a_max=mpc.U(1,1)
       Umax=mpc.U(5,5)

       v,delta=mpc.MPC(X, Xf, Uf, Uk_1, Ufk_1, Uf_nc, 0.02, Np, a_max, Umax, 2.95)

       Uk_1=mpc.U(v,delta)
       Ukf_1=mpc.U(waypoints[index_nowref][3], waypoints[index_nowref][4])
       X.x,X.y,X.theata=carmodel.compute_next_state(X.x, X.y, X.theta, v, delta, L, 0.02)
       print(X.x,X.y,X.theta,v,delta)
