import random

import numpy as np

from random import randint
from random import sample


def getDispatchPriority(n, graph, edgesH, levellist, costList):
    weightList = []
    for i in range(n):
        weightList.append([i, costList[i, 0] * 2 + costList[i, 1]])

    for i in reversed(range(len(levellist))):
        for j in range(levellist[i]):  # 遍历某层
            _id = graph[i, j]
            if _id in edgesH.keys():
                for k in edgesH[_id]:  # {}[_id] # k in list
                    weightList[_id][1] += weightList[k][1]

    weightList = sorted(weightList, key=(lambda x: [x[1], x[0]]), reverse=True)
    # weightList = np.array(weightList,dtype=np.int)[:, 0].tolist()
    tt = []

    for i in weightList:
        tt.append(i[0])

    return tt


def DataMake(n: int, fat, density, CCR):
    # n   任务规模 任务数量
    # fat 图的宽度（ｆａｔ），在相同的任务规模下，较小的 ｆａｔ值可构造较“瘦高”的 ＤＡＧ，ｆａｔ较大时会导致较“矮胖”的 ＤＡＧ生成，
    #    {0.1 0.3 0.5，0.7 0.9}
    # density 边的密度 为｛０．５，０．６，０．７，０．８，０．９}
    # CCR 通信／计算比 为｛０．３，０．４，０．５｝
    MaxWidth = n ** 0.5 * (fat * 2)  # fat 乘 n的根号

    # MinHeight = n ** 0.5 / fat  #至少这么多层
    model = 2
    if MaxWidth <= 1:
        MaxWidth = 1
        model = 2
        # MinHeight = n-1
    ### 将任务随机 不平衡分成n组 空位用-1填充
    graph = np.ones([n, int(MaxWidth)], dtype=np.int) * -1
    nList = list(range(0, n))  # [ 编号    ]变成有n个点 list
    levellist = []
    for i in range(n):
        if len(nList) == 0:
            graph = graph[:i, :]  # 将多余的行去除
            break
        _flag1 = min(randint(1, int(MaxWidth)), len(nList))  # 如果随机生成的数比list的剩下长度大 则只遍历len次
        _flag2 = min(randint(int(MaxWidth / 2), int(MaxWidth)), len(nList))  # 这种设定了一般情况下每层最少为最大的一半
        if model == 1:
            flag = _flag2
        else:
            flag = _flag1
        levellist.append(flag)  # 储存每行有多少个node  list里面有多少个元素就有level
        for j in range(flag):
            graph[i, j] = nList.pop(0)
    # print("levellist",levellist)
    # print("graph_1", graph_1)

    ###建立连接 ：在每层间根据参数建立连接
    # 由于论文没有提到跳级情况（论文指向了另一篇论文 那篇论文提到了这个参数并在参考文献中指向了一个找不到的网址 ） 这里默认不跳
    edgesH = {}  # 下面这个循环后 [ 那两层间的连接 ][前一层第几个元素] [ 连接到了后一层的第几个元素]
    edgesQ = {}
    for i in range(len(levellist) - 1):
        MaxEdge = levellist[i] * levellist[i + 1]
        edgeNum = int(MaxEdge * density)
        if MaxEdge < 1:
            MaxEdge = 1
        if edgeNum < 1:
            edgeNum = 1
        if density == 0:
            edgeNum = 0
        nList = list(range(0, MaxEdge))

        _sample = np.array(sample(nList, edgeNum))
        # edgesH
        _sample1 = np.array(_sample / levellist[i + 1], dtype=np.int)
        _sample2 = _sample % levellist[i + 1]
        a = np.vstack([_sample1, _sample2]).T

        # _dict = {}
        # print("i", i, "sample 1", _sample1)
        for j in np.unique(_sample1):
            tt = a[a[:, 0] == j, 1]
            id = graph[i, j]
            for z in range(tt.shape[0]):
                y = tt[z]
                tt[z] = graph[i + 1, y]
            edgesH[id] = np.sort(tt).tolist()
            # _dict[ j ] = np.sort(tt)
        # edgesH.append(_dict)
        # edgesQ
        a = np.vstack([_sample1, _sample2]).T
        # _dict = {}
        for j in np.unique(_sample2):
            tt = a[a[:, 1] == j, 0]
            ##
            id = graph[i + 1, j]
            for z in range(tt.shape[0]):
                y = tt[z]
                tt[z] = graph[i, y]
            edgesQ[id] = np.sort(tt).tolist()
            # _dict[ j ] = np.sort(tt)
        # edgesQ.append(_dict)
    ###生成任务量
    costList = np.ones([n, 2]) * -1  # 存后驱  # [第几个任务 编号,  0 传输过去需要的时间   1 计算在云端的时间 要本地时*8]
    _sum1 = 0
    _sum2 = 0
    for i in range(n):
        costList[i, 0] = randint(1, 11) * 5 / (128 * 8) * 1000  # 传输过去需要的时间 ms
        costList[i, 1] = randint(1, 11)  # 如果加上
        _sum1 = _sum1 + costList[i, 0]
        _sum2 = _sum2 + costList[i, 1]

    _sum3 = _sum1 / CCR  # 理论上的值
    costList[:, 1] = costList[:, 1] * _sum3 / _sum2
    priorityList = getDispatchPriority(n, graph, edgesH, levellist, costList)
    # 返回
    return graph, levellist, edgesH, edgesQ, costList, priorityList


if __name__ == "__main__":
    graph, levellist, edgesH, edgesQ, costList, priorityList = DataMake(n=25, fat=0.9, density=0.5, CCR=0.5)

    print("graph", "\n", graph)  # 任务的位置 -1为空
    print("levellist", "\n", levellist)  # 每层几个任务
    print("edgesH", "\n", edgesH)  # [ 那两层间的连接 ][前一层第几个元素] [ 连接到了后一层的第几个元素]
    print("edgesQ", "\n", edgesQ)  # [ 那两层间的连接 ][前一层第几个元素] [ 连接到了后一层的第几个元素] #这个可能有问题
    print("costList", "\n", costList)  # 0 任务传输过去/回来 时间消耗  1 任务在本地计算消耗时间
    print("priorityList", "\n", priorityList)  #
