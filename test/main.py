#import sys
#sys.path.append("/home/zyk/mpc/lib")


import numpy as np
# import mpc
import loadwaypoints
import carmodel
import matplotlib.pyplot as plt
import math
import datetime
import  newmpc


# 存储上一时刻控制量
# with open("/home/zyk/mpc/resources/tmpUk_1.txt", 'w') as f:
#        f.write("0 0\n")


# 日志
log = open("/home/zyk/mpc/resources/loggo.txt", 'w')
log.write(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

#时间线
T_control=0.02   # 控制时间间隔
timestemps=np.arange(0,55,T_control)

#路点
# 直线200路点
# waypoints=loadwaypoints.GetRefTrack("/home/zyk/mpc/resources/linewaypoint/line50.xlsx")
# 直线1400路点
waypoints=loadwaypoints.GetJsonRefTrack("/home/zyk/mpc/resources/2021-4-6.json")
#sin2000
# waypoints=loadwaypoints.GetRefTrack("/home/zyk/mpc/resources/linewaypoint/roads.xlsx")


# X=mpc.X(waypoints[0][0],waypoints[0][1],waypoints[0][2])
now=newmpc.Trajectory(waypoints[0][0],waypoints[0][1],waypoints[0][2],0,0)
lastspeed=0
lastdelta=0
lastrefspeed=0
lastrefdelta=0
index_nowref=0
Np=40
L=2.95
index_lastref=1
plt.ion()
for t in timestemps:
       plt.plot(waypoints[:, 0], waypoints[:, 1])
       #根据当前状态和参考路点选择当前参考点及Np个参考点waypoints_nc
       distance = 10000
       (len_waypoints,b)=waypoints.shape
       for i in range(0,len_waypoints):
              if (waypoints[i, 0] - now.x)**2+(waypoints[i, 1] - now.y)**2 < distance:
                     distance = (waypoints[i, 0] - now.x)**2+(waypoints[i, 1] - now.y)**2
                     temp = i
       index_nowref=temp
       log.write("index_lastref:{}".format(index_lastref))
       if index_nowref<index_lastref:
              index_nowref=index_lastref
    #    waypoints_nc=[]
       reflist=[]
       for i in range(Np):
              if index_nowref+i <= len_waypoints-1:
                    points=newmpc.Trajectory(waypoints[index_nowref+i][0],waypoints[index_nowref+i][1],waypoints[index_nowref+i][2],waypoints[index_nowref+i][3],waypoints[index_nowref+i][4])
                    reflist.append(points)
              else:
                    points=newmpc.Trajectory(waypoints[len_waypoints-1][0],waypoints[len_waypoints-1][1],waypoints[len_waypoints][2],waypoints[len_waypoints-1][3],waypoints[len_waypoints-1][4])
       v,delta,a,vf,deltaf=newmpc.MPC(now, reflist,lastspeed,lastdelta,lastrefspeed,lastrefdelta,40)
       plt.plot(waypoints[index_nowref][0],waypoints[index_nowref][1],color="red",marker=".")
       log.write("----------------------------------------------------------------------------------------------------------\n")
       log.write("时刻:{:<5f}   参考点:{}\n".format(t,index_nowref))
       log.write("当前状态: x={:<15f}   y={:<15f}    heading={:<15f}   计算控制量: v={:<15f}   delta={:<15f}\n".format(now.x,now.y,now.phi,v,delta))
       log.write("参考状态: x={:<15f}   y={:<15f}    heading={:<15f}   参考控制量: v={:<15f}   delta={:<15f}\n".format(reflist[0].x,reflist[0].y,reflist[0].phi,reflist[0].v,reflist[0].delta))
    #    Uk_1=mpc.U(v,delta)
    #    Ukf_1=mpc.U(waypoints[index_nowref][3], waypoints[index_nowref][4])
       x,y,heading=carmodel.compute_next_state(now.x,now.y,now.phi,v,delta,T_control,L)
       now.x=x
       now.y=y
       now.phi=heading
       now.v=v
       now.delta=delta
       lastspeed=v
       lastdelta=delta
       lastrefspeed=vf
       lastrefdelta=deltaf
    #    X.x,X.y,X.theata=carmodel.compute_next_state(X.x, X.y, X.theta, v, delta, T_control, L)
       index_lastref=index_nowref
    #    print(X.x,X.y,X.theta,v,delta)
       plt.plot(now.x,now.y,color="black",marker=".")
       plt.show()
       plt.pause(0.001)
       plt.cla()
log.close()