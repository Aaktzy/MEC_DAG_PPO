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

# 定义网络。此网络为actor和critic的结合
dir = "./model"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")
class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(state_dim, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 128)

        self.linear_actor1 = nn.Linear(128, 64)
        self.linear_actor2 = nn.Linear(64, action_dim)  # 输出动作
        self.linear_critic1 = nn.Linear(128, 64)
        self.linear_critic2 = nn.Linear(64, 1)  # 输出Q值

    def actor_forward(self, state, softmax_dim):
        s = F.relu(self.linear_1(state))
        s = F.relu(self.linear_2(s))
        s = F.relu(self.linear_3(s))

        s = F.relu(self.linear_actor1(s))
        prob = F.softmax(self.linear_actor2(s), dim=softmax_dim)  # 输出动作的概率
        return prob

    def critic_forward(self, state):
        s = F.relu(self.linear_1(state))
        s = F.relu(self.linear_2(s))
        s = F.relu(self.linear_3(s))

        s = F.relu(self.linear_critic1(s))
        return self.linear_critic2(s)  # 输出Q值


GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.1
learningRate = 0.0001

n = 30
fat = 0.45
density = 0.3
CCR = 0.5
p = 10

nList = [10, 15, 20, 25, 30, 35, 40, 45, 50]
fatList = [0.3, 0.5, 0.7, 0.9]
densityList = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
CCRList = [0.3, 0.5, 0.7]


# PPO算法
class PPO():
    def __init__(self, state_dim, action_dim):
        # 状态空间和动作空间的维度
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 构建网络
        self.net = Net(self.state_dim, self.action_dim).to(device)
        # 优化器
        self.optim = optim.Adam(self.net.parameters(), learningRate)

        # 创建buffer
        self.data = []

    # 把交互数据存入buffer
    def put_data(self, transition):
        self.data.append(transition)

    # 将数据形成batch
    def make_batch(self):
        list_s = []
        list_a = []
        list_r = []
        list_s_ = []
        list_prob_a = []
        list_done = []

        for transition in self.data:
            s, a, r, s_, prob_a, done = transition
            list_s.append(s)
            list_a.append([a])
            list_r.append([r])
            list_s_.append(s_)
            list_prob_a.append([prob_a])
            done_mask = 0 if done else 1
            list_done.append([done_mask])

        s = torch.tensor(list_s, dtype=torch.float).to(device)
        a = torch.tensor(list_a).to(device)
        r = torch.tensor(list_r, dtype=torch.float).to(device)
        s_ = torch.tensor(list_s_, dtype=torch.float).to(device)
        done_mask = torch.tensor(list_done, dtype=torch.float).to(device)
        prob_a = torch.tensor(list_prob_a).to(device)

        self.data = []  # 清空数组
        return s, a, r, s_, done_mask, prob_a

    def learn(self):
        s, a, r, s_, done_mask, prob_a = self.make_batch()

        for i in range(3):
            # 计算td_error误差，模型目标就是减少td_error
            td_target = r + GAMMA * self.net.critic_forward(s_) * done_mask
            delta = td_target - self.net.critic_forward(s)
            delta = delta.detach().cpu().numpy()

            # 计算advantage，即当前策略比一般策略（baseline）要好多少
            # policy的优化目标就是让当前策略比baseline尽量好，但每次更新时又不能偏离太多，所以后面会有个clip
            list_advantage = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = GAMMA * LAMBDA * advantage + delta_t[0]
                list_advantage.append([advantage])
            list_advantage.reverse()
            advantage = torch.tensor(list_advantage, dtype=torch.float).to(device)

            # 计算ratio，防止单词更新偏离太多
            pi = self.net.actor_forward(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b = exp( log(a) - log(b) )

            # 计算clip，保证ratio在（1-eps_clip, 1+eps_clip)范围内
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage

            # 这里简化ppo，把policy loss和value loss放在一起计算
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.critic_forward(s), td_target.detach())

            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()


def main():
    graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
    env = environment.dagEnv(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
    _priorityList = list(priorityList)
    agent = PPO(len(env.getState()), 2)
    average_reward = 0
    average_l = 0
    # 主循环
    for episode in range(100000):
        graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
        env.reset2(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
        _priorityList = list(priorityList)
        s = env.getState()
        tot_reward = 0
        actions = []
        for j in range(n):
            ####复现策略

            ####
            if len(priorityList):
                id = priorityList.pop(0)
            else:
                id = -1

            s = np.array(s, dtype=float)
            prob = agent.net.actor_forward(torch.from_numpy(s).float().to(device), 0)
            a = int(prob.multinomial(1))
            actions.append([id, a])
            s_, r, done = env.step(id, a)

            s_ = np.array(s_, dtype=float)

            rate = prob[a].item()
            # 保存数据
            agent.put_data((s, a, r, s_, rate, done))
            s = s_
            tot_reward += r

            if done:
                # average_reward = average_reward + 1 / (episode + 1) * (tot_reward - average_reward)
                _tot_reward = float(int(tot_reward * 100)) / 100
                average_reward = average_reward * 0.99 + tot_reward * 0.01
                l = float(int(env.getAllLocal() * 100)) / 100
                average_l = average_l * 0.99 + l * 0.01
                if episode % 100 == 0:
                    print('Episode ', episode, "tot_reward", _tot_reward, ' %: ', (l + _tot_reward) / l, 'allLocal ', l,
                          'allCloud ', env.getAllCloud(), ' average_%: ', (average_l + average_reward) / average_l,
                          ' average_reward: ', average_reward)

                    if ((average_l + average_reward) / average_l) > 0.61:  # 保存好成绩的模型参数
                        import time
                        t = time.strftime("%Y-%m-%d %H-%M", time.localtime())
                        torch.save(agent, dir + "/" + t + ".pkl")

                break
        # Agent.train_net()
        if episode / 10 > 0:
            agent.learn()


main()
