#####
import environment
import dag


def main(n, graph, priorityList, levelList, edgesH, edgesQ, costList, p):
    # 创建倒立摆环境
    best_r = -1000000000000
    env = environment.dagEnv(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
    # 主循环
    nn = 2 ** n
    best_a = 0
    for i in range(nn):
        env.reset2(graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=p)
        actions = []  # 存动作
        _0bNum = bin(i)[2:]  # 去掉0b的头部
        actions = actions + [0] * (n - len(_0bNum))
        for j in range(len(_0bNum)):
            actions.append(int(_0bNum[j]))
        _priorityList = list(priorityList)
        now_r = 0
        a = 9999

        while True:

            if len(_priorityList):
                id = _priorityList.pop(0)
                a = actions.pop(0)
            s_, r, done = env.step(id, a)
            now_r += r

            if done:
                if now_r > best_r:
                    best_r = now_r

                break

    return best_r, env.getAllLocal()


if __name__ == "__main__":
    GAMMA = 0.95
    LAMBDA = 0.95
    EPS_CLIP = 0.1
    learningRate = 0.0001

    n = 4
    fat = 0.45
    density = 0.3
    CCR = 0.5

    p = 10

    graph, levelList, edgesH, edgesQ, costList, priorityList = dag.DataMake(n=n, fat=fat, density=density, CCR=CCR)
    print("次数：", 2 ** n)
    best_r, l = main(n, graph, list(priorityList), levelList, edgesH, edgesQ, costList, p=10)
    print(best_r, (best_r + l) / l)
