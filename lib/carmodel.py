import math
"""
@brief 根据当前状态量和控制量计算下个时刻状态量
@param x y heading  车辆的横坐标、纵坐标和航向角，航向角单位为rad
@param v delta 车辆的速度和前轮转角，前轮转角单位为弧度，注意不是方向盘转角
@param T 差分方程时间间隔，单位为s
"""
def compute_next_state(x, y, heading, v, delta, T, L):
       x = x + v*math.cos(heading) * T
       y = y + v*math.sin(heading) * T
       heading = heading + v* math.tan(delta) / L * T
       return x, y, heading