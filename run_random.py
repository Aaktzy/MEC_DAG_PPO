'''
PPO算法也是Actor-Critic架构，但是与DDPG不同，PPO为on-policy算法，所以不需要设计target网络，也不需要ReplayBuffer，
并且Actor和Critic的网络参数可以共享以便加快学习。
PPO引入了重要度采样，使得每个episode的数据可以被多训练几次（实际的情况中，采样可能非常耗时）从而节省时间，
clip保证的更新的幅度不会太大。
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
#####
import environment
import dag
#定义网络。此网络为actor和critic的结合

n = 50
fat = 0.9
density = 0.3
CCR= 0.5
p = 10


n = 10
fat = 1
density = 0.3
CCR= 0.7
p = 10
#PPO算法

#主函数：简化ppo 这里先交互T_horizon个回合然后停下来学习训练，再交互，这样循环10000次

def main():
    #创建倒立摆环境

    graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
    env = environment.dagEnv(graph, list(priorityList), levelList, edgesH, edgesQ, costList,p=10)

    average_reward = 0
    average_l = 0
    #主循环
    for episode in range(100000):

        priorityList,local = env.reset1(n=n, fat=fat, density=density, CCR=CCR)
        s = env.getState()
        tot_reward = 0
        while True:
            if len(priorityList):
                id = priorityList.pop(0)
            else:
                id = -1
            s = np.array(s,dtype=float)



            ##########
            s_, r, done = env.step(id, random.randint(0, 1))
            ##########
            #s_, r, done = env.step(id, a)

            ## modify the reward



            #保存数据


            tot_reward += r

            if done:
                # average_reward = average_reward + 1 / (episode + 1) * (tot_reward - average_reward)
                tot_reward = float(int(tot_reward * 100)) / 100
                average_reward = average_reward * 0.99 + tot_reward * 0.01
                l = float(int(env.getAllLocal() * 100)) / 100
                average_l = average_l * 0.99 + l * 0.01
                if episode % 100 == 0:
                    print('Episode ', episode, "tot_reward", tot_reward, ' %: ', (l + tot_reward) / l, 'allLocal ', l,
                          'allCloud ', env.getAllCloud(), ' average_%: ', (average_l + average_reward) / average_l,
                          ' average_reward: ', average_reward)
                break
        # Agent.train_net()




main()






