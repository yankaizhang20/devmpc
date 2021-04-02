import xlrd
import json
import numpy as np

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