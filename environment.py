import numpy as np


class dagEnv():  # 目前设定云端就一个vm服务 #和论文不太一样 论文2.2.2说调度一次变一次 而目前实现的变的次数大于k 就相当于要多出等待这个选项
    # 奖励还没写
    def __init__(self, graph, priorityList, levellist, edgesH, edgesQ, costList, p=10):
        self.TrueTime = 0
        # self.timeList =  []# np.zeros_like(graph)*-1
        ######参数表 不变
        self.graph = graph
        self.levellist = levellist
        self.taskNum = sum(levellist)
        self.edgesH = edgesH
        self.edgesQ = edgesQ
        self.costList = costList
        self.priorityList = priorityList  # 顺序
        self.p = p  # 前驱 后驱填充数

        #########运行数据表  存数据
        # [[id,云端跑剩余时间,上传剩余时间 ,下载剩余时间  ]  [] []]
        self.upload = []
        self.download = []
        self.cloud = []
        # [id, 本地跑剩余时间]
        self.loacal = []

        #########状态表 表示id代表的任务在那个状态
        self.waitList = list(self.priorityList)  # []list

        self.runLList = []
        self.runCList1 = []  # 1， 2
        self.runCList2 = []  # 存正在下载的
        self.finishList = []  # 存搞完的id

        # 奖励和是否结束
        self.reward = 0
        self.done = False  # 习惯性的设定了 其实可以不用 但别删因为还是用了
        self.stepNum = 0

    def step(self, id, action):
        ###########
        #####之后尝试可以第一二或3步 不跑直接存到对应的表
        ##########
        #  wait ->  run -> finsh 只存id

        self.reward = 0
        # 检查合法
        None
        self.id = id
        self.action = action
        ###是否准备好？ 好了走后面 不能跑wait
        if action == 0:
            if self.isreadyRun(id, action) == False:  # 本地前驱没准备好
                self.runWait(id, action)
        elif action == 1:  # 云端  前驱跑完了 或上传中到下载中
            if self.isreadyUpload(id, action) == False:  # 能否上传
                self.runWait(id, action)
        else:
            print("error")
        ### 从 wait 删除

        self.waitList.pop(self.waitList.index(id))

        ### 0本地 1云端 #加入对应队列
        if action == 0:  # 本地
            self.loacal.append([id, self.costList[id, 1]])
            self.runLList.append(id)
        elif action == 1:  # 云端
            self.upload.append([id, self.costList[id, 1] / 8, self.costList[id, 0], self.costList[id, 0]])
            self.runCList1.append(id)

        # 能跑后跑一下
        self.runTask()

        # 检查wait是否为空? 是跑end
        if len(self.waitList) == 0:  # 如果id为-1
            self.done = self.runEnd()
        # 最后stepNum+1
        self.stepNum += 1
        # 获取state
        state = self.getState()

        # 返回状态 奖励 done
        return state, self.reward, self.done

    # def runTask(self):  # 加入runlist 的改变
    #
    #
    #     _reward = 0
    #     minT = self._run()
    #     self.TrueTime += minT
    #     _reward -= minT
    #     self.reward = self.reward + _reward
    def runTask(self):
        _falge = True
        _reward = 0
        while _falge == True:
            ###找出最小的时间
            minT = self._run()
            self.TrueTime += minT
            _reward -= minT
            # 判断能否跳出 条件：前面准备好了
            if len(self.loacal) == 0 or len(self.upload):
                break

        self.reward = self.reward + _reward

    def runWait(self, id, action):
        _falge = True
        _reward = 0
        while _falge == True:
            ###找出最小的时间
            minT = self._run()
            self.TrueTime += minT
            _reward -= minT
            # 判断能否跳出 条件：前面准备好了
            if action == 0:
                if self.isreadyRun(id, action) == True:  # 本地前驱准备好了
                    break
            else:  # 云端  前驱跑完了 或上传中到下载中
                if self.isreadyUpload(id, action) == True:  # 能否上传
                    break
        self.reward = self.reward + _reward

    def runEnd(self):
        _falge = True
        _reward = 0
        while _falge == True:
            # 判断能否跳出 条件：跑完全部任务
            if len(self.waitList) == 0 and \
                    len(self.runLList) == 0 and \
                    len(self.runCList1) == 0 and \
                    len(self.runCList2) == 0:
                self.reward = self.reward + _reward
                return True
            ###找出最小的时间 并跑
            minT = self._run()
            self.TrueTime += minT
            _reward -= minT
        return 0  # 彻底跑完的任务 之后加到finish中 #不用加了

    def isreadyRun(self, id, action):
        # 能否开始跑
        _falge = True
        if action == 0:  # 本地
            if id in self.edgesQ.keys():  # 如果前驱存在
                for i in self.edgesQ[id]:
                    if i not in self.finishList:
                        _falge = False
            else:
                None
        elif action == 1:  # 云
            if id in self.edgesQ.keys():  # 如果前驱存在
                for i in self.edgesQ[id]:
                    # 本地没跑完 云端也没跑完
                    if i not in self.finishList and i not in self.runCList2:
                        _falge = False
        return _falge

    def isreadyUpload(self, id, action):
        # 能否上传
        _falge = True
        if action == 1:  # 云
            if id in self.edgesQ.keys():  # 如果前驱存在
                for i in self.edgesQ[id]:

                    # 本地没跑完 云端也没跑完
                    if i not in self.finishList and i not in self.runCList1 and i not in self.runCList2:
                        _falge = False
        else:
            return (-1, "acion error only 1 ")
        return _falge

    def _run(self):
        _timeList = []
        if len(self.loacal) >= 1:
            _timeList.append(self.loacal[0][1])
        if len(self.download) >= 1:
            _timeList.append(self.download[0][3])
        if len(self.cloud) >= 1:
            _timeList.append(self.cloud[0][1])
        if len(self.upload) >= 1:
            _timeList.append(self.upload[0][2])

        if len(_timeList) == 0:
            print(self.id, self.action)
        minT = min(_timeList)

        ### 如果等于最小的时间 则进入下一部分 不等于 时间减少
        if len(self.loacal) >= 1:
            if self.loacal[0][1] == minT:
                _t = self.loacal.pop(0)
                self.runLList.pop(self.runLList.index(_t[0]))
                self.finishList.append(_t[0])
            else:
                self.loacal[0][1] -= minT
        # 处理云部分的 顺序 下载 运行 上传  eg：避免运行进下载，又减了相同的时间
        if len(self.download) >= 1:
            if self.download[0][3] == minT:
                _t = self.download.pop(0)
                self.runCList2.pop(self.runCList2.index(_t[0]))  # 同时删除云列表表示 卸载跑完了
                self.finishList.append(_t[0])
            else:
                self.download[0][3] -= minT
        if len(self.cloud) >= 1:
            if self.cloud[0][1] == minT:
                _t = self.cloud.pop(0)
                self.runCList1.pop(self.runCList1.index(_t[0]))  # 同时删除云列表表示 卸载跑完了
                self.runCList2.append(_t[0])
                self.download.append(_t)
            else:
                self.cloud[0][1] -= minT
        if len(self.upload) >= 1:
            if self.upload[0][2] == minT:
                _t = self.upload.pop(0)
                self.cloud.append(_t)
            else:
                self.upload[0][2] -= minT
        return minT

    def reset1(self, n=20, fat=0.5, density=0.5, CCR=0.9, p=10):  # 输入参数 重设环境
        import dag
        graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
        self.__init__(graph, priorityList, levelList, edgesH, edgesQ, costList, p=p)
        return list(priorityList), self.getAllLocal()

    def reset2(self, graph, priorityList, levelList, edgesH, edgesQ, costList, p=10):  # 输入参数 重设环境
        self.__init__(graph, priorityList, levelList, edgesH, edgesQ, costList, p=p)

    def getState(self):
        #######返回 state
        ###之后可以尝试减少state
        state = []
        ### (每个list的len)
        for i in [self.upload, self.download, self.cloud, self.loacal, self.waitList, self.finishList]:
            state.append(len(i))
        ### (当前list各项跑完剩余时间)
        time = 0
        for i in range(len(self.upload)):
            time += self.upload[i][2]
        state.append(time)
        time = 0
        for i in range(len(self.download)):
            time += self.download[i][3]
        state.append(time)
        time = 0
        for i in range(len(self.cloud)):
            time += self.cloud[i][1]
        state.append(time)
        time = 0
        for i in range(len(self.loacal)):
            time += self.loacal[i][1]
        state.append(time)
        ### (下一个的 前驱 ,后驱) 不够填充-1

        # if self.stepNum< len(self.priorityList): #不是最后的任务
        #
        #     id = self.priorityList[self.stepNum]  # 下个要跑的任务的id
        #
        #     if id in self.edgesQ.keys():# 如果有前驱
        #         if len(self.edgesQ[id])<=self.p:
        #             state = state+ self.edgesQ[id] + [-1]*(self.p - len(self.edgesQ[id]))
        #         else:
        #             state = state + self.edgesQ[id][:self.p]
        #     else:
        #         state = state +[-1] * self.p
        #     if id in self.edgesH.keys():# 如果有后驱
        #         if len(self.edgesH[id]) <= self.p:
        #             state = state+ self.edgesH[id] + [-1]*(self.p - len(self.edgesH[id]))
        #         else:
        #             state = state + self.edgesH[id][:self.p]
        #     else:
        #         state = state +[-1] * self.p
        # else:
        #     state = state +[-1] * self.p
        #     state = state +[-1] * self.p
        ##### 前驱的数量 存入前驱完成计算的数量 # 后驱的数量
        if self.stepNum < len(self.priorityList):  # 不是最后的任务
            id = self.priorityList[self.stepNum]  # 下个要跑的任务的id

            if id in self.edgesQ.keys():  # 如果有前驱
                state.append(len(self.edgesQ[id]))  # 前驱的数量
                # 存入前驱完成计算的数量
                _num = 0
                for i in self.edgesQ[id]:
                    if i in self.finishList or i in self.download:
                        _num += 1
                state.append(_num)
            else:
                state = state + [-1] * 2
            if id in self.edgesH.keys():  # 如果有后驱
                state.append(len(self.edgesH[id]))  # 后驱的数量
            else:
                state = state + [-1]
        else:
            state = state + [-1] * 2
            state = state + [-1]
        # 加入前后驱 总和开销
        if self.stepNum < len(self.priorityList):  # 不是最后的任务
            id = self.priorityList[self.stepNum]  # 下个要跑的任务的id

            if id in self.edgesQ.keys():  # 如果有前驱
                # 存入前驱 全部通讯成本 全部本地计算成本
                _Time_sum1 = 0
                _Time_sum2 = 0
                for i in self.edgesQ[id]:
                    _Time_sum1 = self.costList[i, 0] * 2
                    _Time_sum2 = self.costList[i, 1] * 2
                state.append(_Time_sum1)
                state.append(_Time_sum2)

            else:
                state = state + [0] * 2
            if id in self.edgesH.keys():  # 如果有后驱
                _Time_sum1 = 0
                _Time_sum2 = 0
                for i in self.edgesH[id]:
                    _Time_sum1 = self.costList[i, 0] * 2
                    _Time_sum2 = self.costList[i, 1] * 2
                state.append(_Time_sum1)
                state.append(_Time_sum2)

            else:
                state = state + [0] * 2
        else:
            state = state + [0] * 4
        ### (下一个任务的各项消耗)
        if self.stepNum < len(self.priorityList):  # 不是最后的任务
            localTime = self.costList[id, 1]
            cloudTime = self.costList[id, 1] / 8
            uploadTime = self.costList[id, 0]
            downloadTime = self.costList[id, 0]
            for i in [localTime, cloudTime, uploadTime, downloadTime]:
                state.append(i)
        else:
            state = state + [-1] * 4
        return state

    def getAllLocal(self):
        sum = 0
        for i in range(len(self.costList)):
            sum += self.costList[i, 1]
        return sum

    def getAllCloud(self):
        sum = 0
        for i in range(len(self.costList)):
            sum += self.costList[i, 1] / 8
            sum += self.costList[i, 0] * 2
        return sum
