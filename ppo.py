import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.1


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


class PPO():
    def __init__(self, state_dim, action_dim, learningRate, device):
        # 状态空间和动作空间的维度
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        # 构建网络
        self.net = Net(self.state_dim, self.action_dim).to(device)
        # 优化器
        self.learningRate = learningRate
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

        s = torch.tensor(list_s, dtype=torch.float).to(self.device)
        a = torch.tensor(list_a).to(self.device)
        r = torch.tensor(list_r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(list_s_, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(list_done, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(list_prob_a).to(self.device)

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
            advantage = torch.tensor(list_advantage, dtype=torch.float).to(self.device)

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

    def save(self, dir):
        torch.save(self.net, dir)

    def load(self, dir):
        self.net = torch.load(dir, map_location=self.device)
        self.optim = optim.Adam(self.net.parameters(), self.learningRate)
