# 2020.11.14      by张艳凯 increase： 增加选取距当前位置最近的前置参考点代码，只针对route_1
# 2020.11.15 9:00 by张艳凯 debug：跟新状态量忘记乘时间量0.02了,值得注意的是修改后的并不如之前的效果好，转弯的时候转向不足
# 2020.11.15 9:46 by张艳凯 try: 更改车速限制为3后,相比于0.3的运行结果较好，但是转角值仍然在剧烈变化。当初始位置为0时，转角值依然不是0，找到原因（车辆控制信息文件tmpUK_1.txt运行结束后没有清零）
#            9:54 by张艳凯 try: 联合matlab测试选取最近路点的效果。当初始点为0.5的时候依然会走sin，当初始点在参考点的时候，比较完美地走完美的走完了第一段直线，中间有轻微的转向（肉眼不可见）
# 2020.11.16 8:51 by张艳凯 increase： 增加参考点至所有路段
#                         increase： 增加运行绘制图片前将tmpUk_1文件重置为0 0\n代码
#                         problem： report： else:  #处在第三段则选择大于且最靠近(70,y)的参考点
#                                   TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
#                                   idea： func MPC中设置的全局变量i不能在for循环中使用，直接把for中执行后的if语句穿给n。也不行（UnboundLocalError: local variable 'n' referenced before assignment）
#                                   idea： 是因为n的作用域的原因导致unboundlocalerror，在func MPC开头增加n=0。也不行
#                                   solve： 是把python中的与操作搞错了，&是按位与，and才是判断条件
#                         problem： 第三段直线不能按照预期走，停下不动了
# 2020.11.16 kai15 by张艳凯  increase： 增加参考点显示模块
#                            problem: report: 参考点选择有问题，第一段直线在x=40左右参考点回到（0,0），在二段和第三段也出现参考点回退的情况。
#                                     idea:   第二段末尾922参考点处出现rx>x,但是ry<y的情况，更改选择第二段参考点的策略（rx>x and ry>y）。成了
#2020.11.19 kai15 by刘翀    increase：重写了参考点选取模块，重写了权重矩阵，增加了轨迹显示，实际误差已经小于2cm
#2020.11.20 kai33 by刘翀    change：增加预测时域到20，效果很好
#2020.11.27 newmpc1 by张艳凯  change: 从txt文件中读取参考路点
#2021.3.3 newmpc1 by刘翀  change: 将mpc的预测步数改为自适应，并将预测距离调为可变
# 如果在纯python环境不仿真运行，设置forpython=1，仿真设置=0
forpython = 1
import numpy as np
import math
from cvxopt import solvers as so, matrix
import array
import xlrd
import json

if forpython == 1:
    import logging
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# 从文件中读取上一时刻的控制参考量,该函数需要改进，文件过大时浪费的空间太多
def readU_k_1(testname):
    with open(testname, 'r') as f:
        lines = f.readlines()
        last = lines[-1]
    last = last.strip('\n')
    last = last.split(' ')
    last[0] = float(last[0])
    last[1] = float(last[1])
    return last


# 将当前控制量存入txt文件
def storeU_k_1(testname, v_k_1, delta_k_1):
    with open(testname, 'a') as f:
        f.write("%s %s\n" % (v_k_1, delta_k_1))


class trajectory(object):
    def __init__(self, x, y, heading, delta, accelerate, time, speed):
        self.x = x
        self.y = y
        # 航向角
        self.phi = heading
        # 曲率
        # self.curve = curve
        # 加速度
        self.a = accelerate
        # 时间
        self.t = time
        # 速度
        self.v = speed
        # 前轮转角
        # self.delta = math.asin(2.95 * self.curve)
        self.delta = delta
        # self.Xr = np.array([[self.x, self.y, self.phi]])
        # self.Ur = np.array([[self.v, self.delta]])


# 获取excel中的路点信息
def GetRefTrack(path):
    file = xlrd.open_workbook(path)
    sheet = file.sheets()[0]
    rows = sheet.nrows
    cols = sheet.ncols
    array = np.zeros((rows, cols))
    for x in range(cols):
        collist = sheet.col_values(x)
        col = np.matrix(collist)
        array[:, x] = col
    return array
# 获取json中的路点信息
def GetJsonRefTrack(jsonpath):
    with open(jsonpath,'r') as f:
        data = json.load(f)
        trajectory = data['trajectory']
        list = []
        for i in trajectory:
            heading_rad=i['heading']/180*math.pi
            angel_rad=i['angle']/180*math.pi
            list.append([i['x_point'], i['y_point'], heading_rad, i['speed'], angel_rad])
        array = np.array(list)
    return array

# 矩阵求幂
def Matrixpower(A, n):
    result = A
    if n == 1:
        return result
    for i in range(1, n, 1):
        result = np.dot(result, A)
    return result


def MPC(x, y, heading, t, lastref):  # x,y,heading为当前车辆状态（或者说上一时刻状态)
    # 在仿真条件下 需要弧度角度转化
    if forpython != 1:
        heading = heading / 180 * math.pi
    filename = "D:\\Python\\install\\tmpUk_1.txt"
    enob = 4  # 控制量的小数点后的位数
    # 如果到了选择新的参考点并计算新的控制量的时候则MPC
    # 有的被0.02整除的点也并不能调用mpc,怀疑是最后的小数导致取余为0失败
    t *= 1000
    t = int(t)
    n=0
    # 打开日志文件
    loggo = open("D:\\Python\\install\\loggo.txt", 'a')
    # 每隔0.02s选择一个参考点，到选择参考点的时刻
    if t % 20 == 0:
        # 参考轨迹点值，包括三段，每段包括x,y,heading,v,curve
        # rout = GetRefTrack("D:\\trackpath_sim\\light\\road--y.xlsx")
        rout = GetJsonRefTrack("D:\\trackpath_sim\\light\\jsonfile.json")
        # route_2 = GetRefTrack("D:\\trackpath_sim\\simulation\\rout_info\\typeL\\1300\\second.xlsx")
        # route_3 = GetRefTrack("D:\\trackpath_sim\\simulation\\rout_info\\typeL\\1300\\third.xlsx")
        # rout = np.vstack((route_1, route_2, route_3))
        # (a, b) = rout.shape
        #从refpoint.txt文件中读取参考点信息（x y heading v delta)
        #rout=np.loadtxt("D:\\Python\\install\\result.txt")
        (rout_size, b) = rout.shape
        print(rout.shape)
        # 确定步长 保证预测的距离
        predicted_diatance = 6
        step = round(predicted_diatance / math.sqrt((rout[1, 0] - rout[0, 0])**2+(rout[1, 1] - rout[0, 0])**2))
        step+=5
        print("step", step)
        # 选择当前参考点
        # 参考点超过太多，选择离当前位置最近的参考点，并且Xr>x
        # if x < 50:  # 如果处在第一段和第二段则选择大于且最靠近(x,0)的参考点
        #     for i in range(a):
        #         if rout[i, 0] > x:
        #             n = i
        #             break
        # elif 50<x<70: # 处在第二段，选择rx>x and ry>y的最近参考点
        #     for i in range(a):
        #         if rout[i,0]>x and rout[i,1]>y:
        #             n=i
        #             break
        #
        # else:  # 处在第三段则选择大于且最靠近(70,y)的参考点
        #     i = 922  # 922点处开始走第三段直线
        #     while i < 1300 and rout[i, 1] <= y:
        #         i += 1
        #     if i >= 1300:
        #         i = 1299
        #     n = i
        distance = 10000
        for i in range(rout_size-1):
            if (rout[i, 0] - x)**2+(rout[i, 1] - y)**2 < distance:
                distance = (rout[i, 0] - x)**2+(rout[i, 1] - y)**2
                n = i
        # 选择目标前面的参考点
        # while n < rout_size and rout[n, 0] < x:
        #     n = n+1
        # 防止参考点倒退
        if n <= lastref:
            n = lastref+1
        # 防止参考点越界
        if n > rout_size-1:
            n = rout_size-1
        # n=int(t/20)
        # while rout[n,0] < x:
        #     n+=1
        # plt.plot(rout[n, 0], rout[n, 1], color='red', marker='x')
        print("n", n)
        print(rout[n, 0])
        # 参考点信息
        rx = round(rout[n, 0], enob)
        ry = round(rout[n, 1], enob)
        rheading = round(rout[n, 2], enob)
        rspeed = round(rout[n, 3], enob)
        rcurve = round(rout[n, 4], enob)
        raccelerate = 0
        rtime = 0.02
        T = 0.02
        L = 2.95  # 车辆模型的轴距为2.95
        loggo.write("当前时刻：%f" %(t))
        loggo.write("当前参考点%d   " %(n))
        loggo.write("n参考点数据：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f\n" % (rx, ry, rheading, rspeed, rcurve))
        loggo.write("当前位置x:%f,y:%f,heading:%f\n" % (x, y, heading))
        # loggo("n+1参考：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f" %(rout[n+1,0],rout[n+1,1],rout[n+1,2],rout[n+1,3],rout[n+1,4]))
        # loggo("n+2参考：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f" %(rout[n+2,0],rout[n+2,1],rout[n+2,2],rout[n+2,3],rout[n+2,4]))
        # loggo("n+3参考：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f" %(rout[n+3,0],rout[n+3,1],rout[n+3,2],rout[n+3,3],rout[n+3,4]))
        # loggo("n+4参考：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f" %(rout[n+4,0],rout[n+4,1],rout[n+4,2],rout[n+4,3],rout[n+4,4]))
        # 为参考点设置数据
        ref = trajectory(rx, ry, rheading, rcurve, raccelerate, rtime, rspeed)
        # k-1时刻的控制量 与 参考控制量
        if n == 0:
            v_k_1 = 0
            delta_k_1 = 0
            v_rk_1 = 0
            delta_rk_1 = 0
        else:
            # 从文件中读取上一时刻的控制量
            Uk_1 = readU_k_1(filename)
            v_k_1 = Uk_1[0]
            delta_k_1 = Uk_1[1]
            v_rk_1 = round(rout[n - 1, 3], enob)
            delta_rk_1 = round(rout[n - 1, 4], enob)
        loggo.write("上一时刻控制量及参考控制量：v_k_1=%f  delta_k_1=%f v_rk_1=%f delta_rk_1=%f\n" % (v_k_1, delta_k_1, v_rk_1, delta_rk_1))
        # k 到k+3时刻的参考控制量f
        v_rk = round(rout[n, 3], enob)
        ref_v = []
        ref_delta = []
        print("rout_size", rout_size)
        # n到了最后step步
        if n > rout_size - step:
            for i in range(n, rout_size, 1):
                ref_delta.append(round((rout[i, 4]), enob))
                ref_v.append(round(rout[i, 3], enob))
            for i in range(0, step-rout_size+n, 1):
                ref_delta.append(round((rout[rout_size-1, 4]), enob))
                ref_v.append(round(rout[rout_size-1, 3], enob))
        # if n >= 1900:  # 1283:
        #     for i in range(0, 20, 1):
        #         ref_delta.append(round((rout[n, 4]), enob))
        #         ref_v.append(round(rout[n, 3], enob))
            # delta_rk = round(math.asin(2.95 * rout[n, 4]), enob)
            # v_rk1 = round(rout[n, 3], enob)
            # delta_rk1 = round(math.asin(2.95 * rout[n, 4]), enob)
            # v_rk2 = round(rout[n, 3], enob)
            # delta_rk2 = round(math.asin(2.95 * rout[n, 4]), enob)
            # v_rk3 = round(rout[n, 3], enob)
            # delta_rk3 = round(math.asin(2.95 * rout[n, 4]), enob)



        else:
            for i in range(0, step, 1):
                ref_delta.append(round(rout[n+i, 4], enob))
                ref_v.append(round(rout[n+i, 3], enob))
        # else:
        #     for i in range(0, 20, 1):
        #         ref_delta.append(round(rout[n, 4], enob))
        #         ref_v.append(round(rout[n + i, 3], enob))
            # delta_rk = round(math.asin(2.95 * rout[n, 4]), enob)
            # v_rk1 = round(rout[n + 1, 3], enob)
            # delta_rk1 = round(math.asin(2.95 * rout[n + 1, 4]), enob)
            # v_rk2 = round(rout[n + 2, 3], enob)
            # delta_rk2 = round(math.asin(2.95 * rout[n + 2, 4]), enob)
            # v_rk3 = round(rout[n + 3, 3], enob)
            # delta_rk3 = round(math.asin(2.95 * rout[n + 3, 4]), enob)
        # loggo("控制时域参考控制量：v_rk=%f delta_rk=%f" %(v_rk,delta_rk))
        # loggo("v_rk1=%f delta_rk1=%f" %(round(v_rk1,enob),round(delta_rk1,enob)))
        # loggo("v_rk2=%f delta_rk2=%f" %(round(v_rk2,enob),round(delta_rk2,enob)))
        # loggo("v_rk3=%f delta_rk3=%f" %(round(v_rk3,enob),round(delta_rk3,enob)))
        # A(k) 5*5
        A = np.array([[1, 0, -ref.v * math.sin(ref.phi) * T, math.cos(ref.phi) * T, 0],
                      [0, 1, ref.v * math.cos(ref.phi) * T, math.sin(ref.phi) * T, 0],
                      [0, 0, 1, math.tan(ref.delta) * T / L, (ref.v * T) / (L * math.cos(ref.delta) ** 2)],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        # B(k) 5*2
        B = np.array([[math.cos(ref.phi) * T, 0],
                      [math.sin(ref.phi) * T, 0],
                      [(math.tan(ref.delta) * T) / L, (ref.v * T) / (L * math.cos(ref.delta) ** 2)],
                      [1, 0],
                      [0, 1]])
        # 3*5
        C = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
        # [X(k) U(k-1)]^T 5*1
        Xi = np.array([[x - ref.x],
                       [y - ref.y],
                       [heading - ref.phi],
                       [v_k_1 - v_rk_1],
                       [delta_k_1 - delta_rk_1]])
        # 拼接psi (3*step)*5 [CA CA^2 CA^3 CA^4  ... CA^step]^T
        Psi = np.dot(C, A)
        for i in range(2, step+1, 1):
            Psi = np.vstack((Psi, np.dot(C, Matrixpower(A, i))))
        # 拼接theta (3*step)*(step*2)
        # Psi(k+1)     [CB             0              0                0                  ...     0]
        # Psi(k+2)     [CAB            CB             0                0                  ...     0]
        # Psi(k+3)     [CA^2B          CAB            CB               0                  ...     0]
        # Psi(k+4)     [CA^3B          CA^2B          CAB              CB                 ...     0]
        #                                              ...
        # Psi(k+n)     [CA^(n-1)B      CA^(n-2)B      CA^(n-3)B        C^(n-4)B           ...     0]
        # Psi(k+step)  [CA^(step-1)B   CA^(step-2)B   CA^(step-3)B     CA^(step-4)B       ...     0]
        temporary = np.dot(C, B)
        Theta = np.hstack((np.dot(C, B), np.zeros([3, (step-1)*2])))
        # range(2, step+1, 1) step是预测域
        for i in range(2, step+1, 1):
            CAn = np.dot(C, Matrixpower(A, i - 1))
            temporary = np.hstack((np.dot(CAn, B), temporary))
            Theta_row = np.hstack((temporary, np.zeros([3, (step - i) * 2])))
            Theta = np.vstack((Theta, Theta_row))
        # 求E (3*step)*1
        E = np.dot(Psi, Xi)
        # 求Rho
        Rho = 20
        # Q状态量的权重矩阵
        # d = 100*(x - ref.x)**2+100*(y - ref.y)**2+2
        d = 100 * (x - ref.x) ** 2 + 100 * (y - ref.y) ** 2+2
        dx = 0.5+100*(x - ref.x)**2
        dy = 0.5+100*(y - ref.y)**2
        # Q (3*step)*(3*step)
        Q = np.zeros((3*step, 3*step))
        for i in range(0, 3*step, 3):
            Q[i][i] = 1*dx
            Q[i+1][i+1] = 1*dy
            Q[i+2][i+2] = 1/d

        # Q = np.array([[1*dx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 1*dy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 1/d, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 1*dx, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 1*dy, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 1/d, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 1*dx, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 1*dy, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 1/d, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1*dx, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1*dy, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/d]])
        # R控制增量的权重矩阵 (2*step)*(2*step)
        R = np.zeros((2*step, 2*step))
        for i in range(0, 2*step, 1):
            # R[i][i] = 0.05
            R[i][i]=0.5

        # R = np.array([[0.05, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0.05, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0.06, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0.06, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0.05, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0.05, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0.05, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0.05]])
        # H [[ θ^T*Q*θ+R 0]
        #    [ 0          ρ] ]
        # temporary=Theta^T*Q*Theta + R
        temporary = np.dot(Theta.T, Q)
        temporary = np.dot(temporary, Theta)
        temporary = temporary + R
        H1 = np.hstack((temporary, np.zeros([2*step, 1])))
        H2 = np.hstack((np.zeros([1, 2*step]), np.array([[Rho]])))
        H = np.vstack((H1, H2)) * 0.5
        # 求f
        f = 2 * np.hstack((np.dot(np.dot(E.T, Q), Theta), np.array([[0]])))

        # 约束求解
        # alpha + ur (2*step)*1
        alpha_ur = np.zeros([2*step, 1])
        print("ref_v.shape", len(ref_v))
        for i in range(0, step, 1):
            print("i= ", i)
            print("alpha_ur[i*2][0]", alpha_ur[i*2][0])
            print("ref_v", ref_v)
            print("ref_v[", i, "]",ref_v[i])
            alpha_ur[i*2][0] = v_k_1 - v_rk_1 + ref_v[i]
            alpha_ur[i*2+1][0] = delta_k_1 - delta_rk_1 + ref_delta[i]
        # alpha_ur = np.array([[v_k_1 - v_rk_1 + v_rk],
        #                      [delta_k_1 - delta_rk_1 + delta_rk],
        #                      [v_k_1 - v_rk_1 + v_rk1],
        #                      [delta_k_1 - delta_rk_1 + delta_rk1],
        #                      [v_k_1 - v_rk_1 + v_rk2],
        #                      [delta_k_1 - delta_rk_1 + delta_rk2],
        #                      [v_k_1 - v_rk_1 + v_rk3],
        #                      [delta_k_1 - delta_rk_1 + delta_rk3]])
        # Uk-Ur-1 (2*step)*1
        urk_urk_1 = np.zeros([2*step, 1])
        urk_urk_1[0][0] = - v_rk_1 + ref_v[0]
        urk_urk_1[0 + 1][0] = - delta_rk_1 + ref_delta[0]
        for i in range(1, step, 1):
            urk_urk_1[i*2][0] = -ref_v[i-1] + ref_v[i]
            urk_urk_1[i*2+1][0] = -ref_delta[i-1] + ref_delta[i]
        # urk_urk_1 = np.array([[- v_rk_1 + v_rk],
        #                       [- delta_rk_1 + delta_rk],
        #                       [- v_rk + v_rk1],
        #                       [- delta_rk + delta_rk1],
        #                       [- v_rk1 + v_rk2],
        #                       [- delta_rk1 + delta_rk2],
        #                       [- v_rk2 + v_rk3],
        #                       [- delta_rk2 + delta_rk3]])
        # Umax
        UV = 1.3
        Udelta = math.pi/4
        U_max = np.zeros([2*step, 1])
        for i in range(0, 2*step, 2):
            U_max[i][0] = UV
            U_max[i+1][0] = Udelta
        # U_max = np.array([[UV],
        #                   [Udelta],
        #                   [UV],
        #                   [Udelta],
        #                   [UV],
        #                   [Udelta],
        #                   [UV],
        #                   [Udelta]])
        # Umin
        U_min = np.zeros([2*step, 1])
        for i in range(0, 2*step, 2):
            U_min[i][0] = -UV
            U_min[i + 1][0] = -Udelta
        # U_min = np.array([[-UV],
        #                   [-Udelta],
        #                   [-UV],
        #                   [-Udelta],
        #                   [-UV],
        #                   [-Udelta],
        #                   [-UV],
        #                   [-Udelta]])
        # Uamax
        maxa = math.pi
        mina = -math.pi
        deltau = math.pi/6
        U_a_max = np.zeros([2*step, 1])
        for i in range(0, 2*step, 2):
            U_a_max[i][0] = maxa
            U_a_max[i+1][0] = deltau
        # 增大前轮转角加速度会减缓走sin曲线形式
        # U_a_max = np.array([[maxa],
        #                     [deltau],
        #                     [maxa],
        #                     [deltau],
        #                     [maxa],
        #                     [deltau],
        #                     [maxa],
        #                     [deltau]])
        # Uamin
        U_a_min = np.zeros([2*step, 1])
        for i in range(0, 2*step, 2):
            U_a_min[i][0] = mina
            U_a_min[i + 1][0] = -deltau
        # U_a_min = np.array([[mina],
        #                     [-deltau],
        #                     [mina],
        #                     [-deltau],
        #                     [mina],
        #                     [-deltau],
        #                     [mina],
        #                     [-deltau]])
        # 求【∆U ε】
        # P、q、G、h、A、b   1/2 *X^t * P * X + q^T * X ,X是列向量,q是列向量
        # cvxopt 必须使用cvxopt的matrix，并且直接在内部写按照列来，所以正常的矩阵到这要转置
        # 但是转换外部写好的矩阵时，是按行来
        H = matrix(H)
        f = matrix(f)
        # matrix里区分int和double，所以数字后面都需要加小数点
        # G是不等式的系数矩阵 [2*step, 2*step+1] 前一半限制控制量本身，后一半限制控制量的增量
        # G1 G2 限制控制量  G3 G4限制控制量增量
        G1 = np.zeros([2*step, 2*step+1])
        G2 = np.zeros([2*step, 2*step+1])
        G3 = np.zeros([2*step, 2*step+1])
        G4 = np.zeros([2*step, 2*step+1])
        for i in range(0, 2*step, 1):
            for j in range(0, i+1, 1):
                # 行为偶数
                if i % 2 == 0:
                    # 列为偶数
                    if j % 2 == 0:
                        G1[i][j] = 1
                        G2[i][j] = -1
                # 行为奇数
                else:
                    # 列为奇数
                    if j % 2 != 0:
                        G1[i][j] = 1
                        G2[i][j] = -1
        for i in range(0, 2*step, 1):
            G3[i][i] = 1
            G4[i][i] = -1
        G = np.vstack((G1, G2))
        G = np.vstack((G, G3))
        G = np.vstack((G, G4))


        # G = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        #               [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        #               [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        #               [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        #               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #               [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]])
        # h是不等式的结果 (2*2*2*step)*1
        h = np.vstack((U_max - alpha_ur, -U_min + alpha_ur))
        h1 = np.vstack((T * U_a_max - urk_urk_1, urk_urk_1 - T * U_a_min))
        h = np.vstack((h, h1))

        # 将h转为行数组
        h = h.flatten()
        # 将h、G转化为matrix
        h = matrix(h)
        G = matrix(G)
        sol = so.qp(H, f.T, G, h)  # 调用优化函数solvers.qp求解
        # loggo(sol['x'])  # 打印结果，sol里面还有很多其他属性，读者可以自行了解

        # 计算控制量u
        u_v = round(sol['x'][0] + (v_k_1 - v_rk_1) + ref.v, enob)
        u_delta = round(sol['x'][1] + (delta_k_1 - delta_rk_1) + ref.delta, enob)
        # u_v1 = sol['x'][2] + v_rk1 + u_v - v_rk
        u_accelerate = round((u_v - v_k_1) / T, enob)
        storeU_k_1(filename, u_v, u_delta)
        loggo.write("speed :%f   " %(u_v))
        loggo.write("angle :%f   " %(u_delta))
        loggo.write("accelerate :%f\n" %(u_accelerate))
        loggo.write("---------------------------------------------------------------------------------------------------------------\n\n")
        loggo.close()
        if forpython:
            return array.array('d', [round(u_v, enob), round(u_delta, enob),n]),n
        else:
            return array.array('d', [round(u_v, enob), round(u_delta, enob), n])
        # return array.array('d', [1, 0])
    # 如果在时间步之间，则直接取上一时刻的控制量
    else:
        Uk_1 = readU_k_1(filename)
        if forpython:
            return array.array('d', [Uk_1[0], Uk_1[1],n]),n
        else:
            return array.array('d',[Uk_1[0],Uk_1[1],n])
        # return array.array('d', [1, 0])
    # 测试从文件中读取最后一行和添加数据
    # Ulist=readU_k_1(filename)
    # loggo(Ulist)
    # loggo(Ulist[0],Ulist[1])
    # storeU_k_1(filename,2,6)


if __name__ == '__main__':
    # route_1 = GetRefTrack("D:\\trackpath_sim\\simulation\\rout_info\\typeL\\1300\\first.xlsx")
    # route_2 = GetRefTrack("D:\\trackpath_sim\\simulation\\rout_info\\typeL\\1300\\second.xlsx")
    # route_3 = GetRefTrack("D:\\trackpath_sim\\simulation\\rout_info\\typeL\\1300\\third.xlsx")
    #rout = np.loadtxt("D:\\Python\\install\\result.txt")
    rout = GetJsonRefTrack("D:\\trackpath_sim\\light\\jsonfile.json")
    # 每次运行之前将tmpUk_1.txt文件重置
    with open("D:\\Python\\install\\tmpUk_1.txt", 'w') as f:
        f.write("0 0\n")
    loggotxt = open("D:\\Python\\install\\loggo.txt", 'w')
    loggotxt.close()
    plt.ion()
    L = 2.95
    x = rout[0][0]
    y = rout[0][1]
    heading = rout[0][2]
    X = []
    Y = []
    T = np.arange(0, 120, 0.5)
    lastref = 0
    for t in T:
        u, nowref = MPC(x, y, heading, t, lastref)
        print("nowref", nowref)
        lastref = nowref
        x = x + u[0]*math.cos(heading) * 0.02
        # if y > rout[nowref,2]:
        y = y + u[0]*math.sin(heading) * 0.02
        # else:
        #     y = y + u[0] / math.cos(u[1]) * math.sin(heading + u[1]) * 0.02
        heading = heading + u[0] * math.tan(u[1]) / L * 0.02
        X.append(x)
        Y.append(y)
        plt.plot(x, y, color="black", marker="*")
        plt.plot(X, Y, color="black", marker=".")
        # n = int(u[2])
        # loggo("参考的点：%s" %(n))
        # plt.plot(rout[n, 0], rout[n, 1], color='red', marker='x')
        plt.plot(rout[:, 0], rout[:, 1])
        plt.show()
        plt.pause(0.001)
        plt.cla()
    plt.ioff()
    plt.plot(X, Y, color="black", marker=".")
    plt.plot(rout[:, 0], rout[:, 1])
    plt.show()


