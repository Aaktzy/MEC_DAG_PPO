import numpy as np
import torch
import random
import time
#####
import environment
import dag
from ppo import PPO

###
dir = "./model/more"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.1
learningRate = 0.0001

n = 50
fat = 0.45
density = 0.3
CCR = 0.5
p = 10

nList = [10, 15, 20, 25, 30, 35, 40, 45, 50]
fatList = [0.3, 0.5, 0.7]
densityList = [0.5, 0.6, 0.7, 0.8, 0.9]
CCRList = [0.3, 0.5, 0.7]


# p = 10

def main():
    record = []
    rl_r = []
    local_r = []
    cloud_r = []
    random_sdf = []
    ##创建环境 下面这几个参数不用管 初始话用一下 之后从list中取
    # 初始化用的数据
    n = 50
    fat = 0.45
    density = 0.3
    CCR = 0.5
    p = 10
    graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)

    # 初始化环境 网络 参数
    env = environment.dagEnv(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
    envR = environment.dagEnv(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
    agent = PPO(len(env.getState()), 2, learningRate, device)
    all_r_reward = 0
    all_reward = 0
    all_l_reward = 0
    all_c_reward = 0
    # 主循环
    for episode in range(1, 4000):
        n = nList[random.randint(0, len(nList) - 1)]
        fat = fatList[random.randint(0, len(fatList) - 1)]
        density = densityList[random.randint(0, len(densityList) - 1)]
        CCR = CCRList[random.randint(0, len(CCRList) - 1)]
        graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
        _priorityList = list(priorityList)
        env.reset2(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p)
        envR.reset2(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p)
        s = env.getState()
        tot_reward = 0
        r_tot_reward = 0

        while True:
            if len(priorityList):
                id = priorityList.pop(0)
            else:
                id = -1
            s = np.array(s, dtype=float)
            prob = agent.net.actor_forward(torch.from_numpy(s).float().to(device), 0)
            a = int(prob.multinomial(1))

            s_, r, done = env.step(id, a)
            _, random_r, _ = envR.step(id, random.randint(0, 1))
            s_ = np.array(s_, dtype=float)

            rate = prob[a].item()
            # 保存数据
            agent.put_data((s, a, r, s_, rate, done))
            s = s_
            tot_reward += r
            r_tot_reward += random_r

            if done:
                # average_reward = average_reward + 1 / (episode + 1) * (tot_reward - average_reward)
                l = env.getAllLocal()
                c = env.getAllCloud()
                all_reward += tot_reward
                all_r_reward += r_tot_reward
                all_l_reward += l
                all_c_reward += c
                average_reward = all_reward / episode
                r_average_reward = all_r_reward / episode
                average_l = all_l_reward / episode
                average_c = all_c_reward / episode
                # print(average_reward,r_average_reward,average_l)
                if episode % 1 == 0:
                    record.append(average_reward)
                    print(average_reward, r_average_reward, average_l, l)
                    print('Episode:', episode, 'ppo优化百分比: ', (l + tot_reward) / l, ' average_%: ',
                          (average_l + average_reward) / average_l, ' average_r: ', average_reward)
                    print("         ", "random优化百分比:", (l + r_tot_reward) / l, ' average_r_%: ',
                          (average_l + r_average_reward) / average_l, ' average_r: ', r_average_reward)
                    print("         优势 ",
                          ((average_l + average_reward) / average_l) / ((average_l + r_average_reward) / average_l))
                    if ((average_l + average_reward) / average_l) > 0.55:  # 保存好成绩的模型参数
                        print("save")
                        t = time.strftime("%Y-%m-%d %H-%M", time.localtime())
                        torch.save(agent, dir + "/bb" + t + ".pkl")
                        # if n <= 20:  # 同时测试任务量 <=20的最优解
                        #     import getBest
                        #     best_r, _ = getBest.main(n, graph, list(_priorityList), levelList, edgesH, edgesQ, costList,p)
                        #     print("最优解", best_r, "优化量占可优化的比例：", (l + tot_reward) / (l + best_r))
                if episode > 2000:
                    rl_r.append(-average_reward)
                    random_sdf.append(-r_average_reward)
                    local_r.append(average_l)
                    cloud_r.append(average_c)
                break
        if episode / 10 > 0:
            agent.learn()

    import matplotlib.pyplot as plt
    plt.plot(record, linewidth=4)
    plt.title("The training process of PPO", fontsize=24)
    plt.xlabel("episode", fontsize=14)
    plt.ylabel("reward", fontsize=14)
    plt.show()

    plt.plot(rl_r, color='green', label='PPO')
    plt.plot(random_sdf, color='blue', label='random')
    plt.plot(local_r, color='red', label='all local')
    plt.plot(cloud_r, color='skyblue', label='all cloud')
    plt.legend()
    plt.xlabel("episode", fontsize=14)
    plt.ylabel("cost time", fontsize=14)
    plt.show()


main()
