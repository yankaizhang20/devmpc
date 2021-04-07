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

class Trajectory(object):
    def __init__(self, x, y, heading, speed, angle):
        self.x = x
        self.y = y
        # 航向角
        self.phi = heading
        # 速度
        self.v = speed
        # 前轮转角
        self.delta = angle




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
    with open(jsonpath, 'r') as f:
        data = json.load(f)
        trajectory = data['trajectory']
        list = []
        for j in range(0, round(len(trajectory))):
            ref = Trajectory(trajectory[j]['x_point'], trajectory[j]['y_point'],
                             trajectory[j]['heading'] * math.pi / 180,
                             trajectory[j]['speed'] * 0.277777778, trajectory[j]['angle'] * math.pi / 180)
            list.append(ref)

    return list
def GetJsonCar(jsonpath):
    with open(jsonpath, 'r') as f:
        data = json.load(f)
        car = data['car']
        x = car['rear_wheel_x']
        y = car['rear_wheel_y']
        heading = car['heading']/180*math.pi
        speed = car['speed']* 0.277777778
        angle = car['angle'] * math.pi / 180

    return x, y, heading, speed, angle
# 矩阵求幂
def Matrixpower(A, n):
    result = A
    if n == 1:
        return result
    for i in range(1, n, 1):
        result = np.dot(result, A)
    return result

def MPC(now, reflist,lastspeed,lastdelta,lastrefspeed,lastrefdelta,step):
    """

    :param now: 现在K时刻的状态
    :param reflist: 参考轨迹，从k到k+step
    :param lastspeed: k-1时刻的速度
    :param lastdelta: k-1时刻的前轮转角
    :param lastrefspeed: k-1时刻的参考速度
    :param lastrefdelta: k-1时刻的参考前轮转角
    :param step: 预测的步数
    :return: 速度，前轮转角，加速度，参考速度，参考前轮转角
    """
    logger.info("last速度为 %s last转角为%s step = %s",lastspeed,lastdelta ,step)
    if step > len(reflist):
        step = len(reflist)
    # heading = heading / 180 * math.pi
    # enob = 4  # 控制量的小数点后的位数
    T = 0.7
    L = 2.95  # 车辆模型的轴距为2.4
    # 当前车辆的状态k
    x = now.x
    y = now.y
    heading = now.phi
    logger.info("x= %s y= %s heading=%s speed= %s", x, y, heading,now.v)
    # 为参考点设置数据
    ref = reflist[0]
    # k-1时刻的控制量 与 参考控制量

    v_k_1 = lastspeed
    delta_k_1 = lastdelta
    #@zyk v_rk_1 = round(lastrefspeed, enob)
    v_rk_1=lastrefspeed
    #@zyk delta_rk_1 = round(lastrefdelta, enob)
    delta_rk_1=lastrefdelta
    # print("上一时刻控制量及参考控制量：v_k_1=%f  delta_k_1=%f v_rk_1=%f delta_rk_1=%f" % (v_k_1, delta_k_1, v_rk_1, delta_rk_1))
    # v_rk = round(now.v, enob)
    v_rk=now.v
    ref_v = []
    ref_delta = []
    for i in range(0, step, 1):
        ref_delta.append(reflist[i].delta)
        ref_v.append(reflist[i].v)
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
    for i in range(2, step + 1, 1):
        Psi = np.vstack((Psi, np.dot(C, Matrixpower(A, i))))
    # 拼接theta (3*step)*(step*2)
    # Psi(k+1)     [CB             0              0                0                  ...     0]
    # Psi(k+2)     [CAB            CB             0                0                  ...     0]
    # Psi(k+3)     [CA^2B          CAB            CB               0                  ...     0]
    # Psi/(k+4)     [CA^3B          CA^2B          CAB              CB                 ...     0]
    #                                              ...
    # Psi(k+n)     [CA^(n-1)B      CA^(n-2)B      CA^(n-3)B        C^(n-4)B           ...     0]
    # Psi(k+step)  [CA^(step-1)B   CA^(step-2)B   CA^(step-3)B     CA^(step-4)B       ...     0]
    temporary = np.dot(C, B)
    Theta = np.hstack((np.dot(C, B), np.zeros([3, (step - 1) * 2])))
    # range(2, step+1, 1) step是预测域
    for i in range(2, step + 1, 1):
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
    d = 100 * (x - ref.x) ** 2 + 100 * (y - ref.y) ** 2 + 2
    dx = 0.5 + 100 * (x - ref.x) ** 2
    dy = 0.5 + 100 * (y - ref.y) ** 2
    # Q (3*step)*(3*step)
    Q = np.zeros((3 * step, 3 * step))
    for i in range(0, 3 * step, 3):
        Q[i][i] = 1 * dx
        Q[i + 1][i + 1] = 1 * dy
        Q[i + 2][i + 2] = 3*(dy+dx)/8

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
    R = np.zeros((2 * step, 2 * step))
    for i in range(0, 2 * step, 1):
        # R[i][i] = 0.05
        R[i][i] = 3*(dy+dx)/10

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
    H1 = np.hstack((temporary, np.zeros([2 * step, 1])))
    H2 = np.hstack((np.zeros([1, 2 * step]), np.array([[Rho]])))
    H = np.vstack((H1, H2)) * 2
    # 求f
    f = 2 * np.hstack((np.dot(np.dot(E.T, Q), Theta), np.array([[0]])))

    # 约束求解
    # alpha + ur (2*step)*1
    alpha_ur = np.zeros([2 * step, 1])
    for i in range(0, step, 1):
        alpha_ur[i * 2][0] = v_k_1 - v_rk_1 + ref_v[i]
        alpha_ur[i * 2 + 1][0] = delta_k_1 - delta_rk_1 + ref_delta[i]
    # alpha_ur = np.array([[v_k_1 - v_rk_1 + v_rk],
    #                      [delta_k_1 - delta_rk_1 + delta_rk],
    #                      [v_k_1 - v_rk_1 + v_rk1],
    #                      [delta_k_1 - delta_rk_1 + delta_rk1],
    #                      [v_k_1 - v_rk_1 + v_rk2],
    #                      [delta_k_1 - delta_rk_1 + delta_rk2],
    #                      [v_k_1 - v_rk_1 + v_rk3],
    #                      [delta_k_1 - delta_rk_1 + delta_rk3]])
    # Uk-Ur-1 (2*step)*1
    urk_urk_1 = np.zeros([2 * step, 1])
    urk_urk_1[0][0] = - v_rk_1 + ref_v[0]
    urk_urk_1[0 + 1][0] = - delta_rk_1 + ref_delta[0]
    for i in range(1, step, 1):
        urk_urk_1[i * 2][0] = -ref_v[i - 1] + ref_v[i]
        urk_urk_1[i * 2 + 1][0] = -ref_delta[i - 1] + ref_delta[i]
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
    Udelta = 0.614348915384996
    U_max = np.zeros([2 * step, 1])
    for i in range(0, 2 * step, 2):
        U_max[i][0] = UV
        U_max[i + 1][0] = Udelta
    # U_max = np.array([[UV],
    #                   [Udelta],
    #                   [UV],
    #                   [Udelta],
    #                   [UV],
    #                   [Udelta],
    #                   [UV],
    #                   [Udelta]])
    # Umin
    U_min = np.zeros([2 * step, 1])
    for i in range(0, 2 * step, 2):
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
    maxa = 0.2
    mina = -0.2
    deltau = math.pi / 4
    U_a_max = np.zeros([2 * step, 1])
    for i in range(0, 2 * step, 2):
        U_a_max[i][0] = maxa
        U_a_max[i + 1][0] = deltau
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
    U_a_min = np.zeros([2 * step, 1])
    for i in range(0, 2 * step, 2):
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
    G1 = np.zeros([2 * step, 2 * step + 1])
    G2 = np.zeros([2 * step, 2 * step + 1])
    G3 = np.zeros([2 * step, 2 * step + 1])
    G4 = np.zeros([2 * step, 2 * step + 1])
    for i in range(0, 2 * step, 1):
        for j in range(0, i + 1, 1):
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
    for i in range(0, 2 * step, 1):
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

    # 计算控制量u
    #@zyk u_v = round(sol['x'][0] + (v_k_1 - v_rk_1) + ref.v, enob)
    u_v=sol['x'][0] + (v_k_1 - v_rk_1) + ref.v
    #@zyk u_delta = round(sol['x'][1] + (delta_k_1 - delta_rk_1) + ref.delta, enob)
    u_delta=sol['x'][1] + (delta_k_1 - delta_rk_1) + ref.delta
    # u_v1 = sol['x'][2] + v_rk1 + u_v - v_rk
    # u_accelerate = round((u_v - v_k_1) / T, enob)
    u_accelerate=(u_v - v_k_1) / T
    # print("speed :", u_v, v_k_1)
    # print("angle :", u_delta)
    # print("accelerate :", u_accelerate)
    # logger.info("速度为 %s 前轮转角为%s 前一时刻前轮转角%s 加速度为%s 参考速度速度为%s 参考前轮转角为%s", round(u_v, enob),
                # round(u_delta, enob), round(delta_k_1, enob), round(u_accelerate, enob), ref.v, ref.delta)
    # 要返回参考控制量返回给idea
    #@zyk return round(u_v, enob), round(u_delta, enob), round(u_accelerate, enob), ref.v, ref.delta
    return u_v,u_delta,u_accelerate,ref.v,ref.delta
    # return array.array('d', [1, 0])

def MPC_wrap(x, y, heading,speed,angle, t, lastref, calculatecount):  # x,y,heading为当前车辆状态（或者说上一时刻状态)
    # 在仿真条件下 需要弧度角度转化
    if forpython != 1:
        heading = heading / 180.0 * math.pi
    filename = "tmpUk_1.txt"

    # 如果到了选择新的参考点并计算新的控制量的时候则MPC

    n=0
    # 打开日志文件
    loggo = open("loggo.txt", 'a')
    # 每隔0.02s选择一个参考点，到选择参考点的时刻

    reflist = GetJsonRefTrack("/home/zyk/mpc/test/jsonfile.json")

    # 确定步长 保证预测的距离
    predicted_diatance = 3
    step = round(predicted_diatance / math.sqrt((reflist[1].x - reflist[0].x)**2+(reflist[1].y - reflist[0].y)**2))
    if step > len(reflist):
        step = len(reflist)
    if step > 250:
        step = 250
    if step < 20:
        step = 20

    print("step", step)

    distance = 10000
    for i in range(len(reflist)-1):
        if (reflist[i].x - x)**2+(reflist[i].y - y)**2 < distance:
            distance = (reflist[i].x - x)**2+(reflist[i].y - y)**2
            n = i
    # 选择目标前面的参考点
    while n < len(reflist) and reflist[n].x < x+0.1:
        n = n+1
    # 防止参考点倒退
    if n <= lastref:
        n = lastref
    # 防止参考点越界
    if n > len(reflist)-1:
        n = len(reflist)-1

    # 参考点信息
    rx = reflist[n].x
    ry = reflist[n].y
    rheading = reflist[n].phi
    rspeed = reflist[n].v
    rcurve =reflist[n].delta
    loggo.write("当前时刻：%f" %(t))
    loggo.write("当前参考点%d   " %(n))
    loggo.write("n参考点数据：rx=%f  ry=%f  rheading=%f  rspeed=%f  rcurve=%f\n" % (rx, ry, rheading, rspeed, rcurve))
    loggo.write("当前位置x:%f,y:%f,heading:%f\n" % (x, y, heading))

    # k-1时刻的控制量 与 参考控制量
    if calculatecount == 1:
        v_k_1 = 0
        delta_k_1 = 0
        v_rk_1 = 0
        delta_rk_1 = 0
    else:
        # 从文件中读取上一时刻的控制量
        Uk_1 = readU_k_1(filename)
        v_k_1 = Uk_1[0]
        delta_k_1 = Uk_1[1]
        v_rk_1 = Uk_1[0]
        delta_rk_1 = Uk_1[1]
    loggo.write("上一时刻控制量及参考控制量：v_k_1=%f  delta_k_1=%f v_rk_1=%f delta_rk_1=%f\n" % (v_k_1, delta_k_1, v_rk_1, delta_rk_1))
    now = Trajectory(x, y, heading, speed, angle)
    speed, angle, accelerate, refv, refangle=MPC(now, reflist[n:], v_k_1, delta_k_1, v_rk_1, delta_rk_1, step)
    loggo.write("speed :%f   " % speed)
    loggo.write("angle :%f   " % angle)
    loggo.write("accelerate :%f\n" % accelerate)
    loggo.write(
        "---------------------------------------------------------------------------------------------------------------\n\n")
    storeU_k_1(filename, speed, angle)
    return [speed, angle], n



if __name__ == '__main__':

    reflist = GetJsonRefTrack("/home/zyk/mpc/test/jsonfile.json")
    # 每次运行之前将tmpUk_1.txt文件重置
    with open("tmpUk_1.txt", 'w') as f:
        f.write("0 0\n")
    with open("loggo.txt", 'w') as f:
        f.write("0 0\n")
    # loggotxt = open("loggo.txt", 'w')
    # loggotxt.close()
    plt.ion()
    L = 2.95
    x, y, heading, speed, angle = GetJsonCar("/home/zyk/mpc/test/jsonfile.json")
    X = []
    Y = []
    timegap = 0.5
    T = np.arange(0, 1000000, timegap)
    lastref = 0
    calculatecount = 1
    for t in T:
        u, nowref = MPC_wrap(x, y, heading,speed, angle, t, lastref, calculatecount)
        calculatecount = calculatecount+1
        print("nowref", nowref)
        lastref = nowref
        heading = heading + u[0] * math.tan(u[1]) / L * timegap
        x = x + u[0]*math.cos(heading) * timegap
        # if y > rout[nowref,2]:
        y = y + u[0]*math.sin(heading) * timegap
        # else:
        #     y = y + u[0] / math.cos(u[1]) * math.sin(heading + u[1]) * 0.02
        # heading = heading + u[0] * math.tan(u[1]) / L * timegap
        X.append(x)
        Y.append(y)
        carx = np.linspace(x - 2, x + 2, 3)
        cary = math.tan(heading) * carx + y - math.tan(heading) * x
        plt.plot(carx, cary, color="green", marker=".")
        plt.plot(x, y, color="black", marker="*")
        plt.plot(X, Y, color="black", marker=".")
        plt.plot(reflist[nowref].x, reflist[nowref].y, color="red", marker="*")
        # n = int(u[2])
        # loggo("参考的点：%s" %(n))
        # plt.plot(rout[n, 0], rout[n, 1], color='red', marker='x')
        for i in range(0,len(reflist)):
            plt.plot(reflist[i].x, reflist[i].y,color="blue", marker=".")
        plt.show()
        plt.pause(0.001)
        plt.cla()
    plt.ioff()
    plt.plot(X, Y, color="black", marker=".")
    for i in range(0, len(reflist)):
        plt.plot(reflist[i].x, reflist[i].y)
    plt.show()


