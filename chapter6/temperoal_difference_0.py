import numpy as np
import matplotlib.pyplot as plt

n_states = 5 # 状态数: A, B, C, D, E（编号1到5）
alpha = 0.1 # 学习率
gamma = 1.0 # 折扣因子
n_episodes = 200 # 回合数

# 初始化价值函数
V = np.zeros(n_states + 2) # 初始化价值函数, 编号0和6为终止状态，1-5为有效状态
true_V = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]) # 真实价值函数(包含终止状态)

V_history = []
np.random.seed(42) # 固定随机种子，便于复现
for eposide in range(n_episodes):
    state = np.random.randint(1, n_states + 1) # 随机选择一个状态
    while True:
        action = np.random.choice([-1, 1]) # 随机选择一个动作
        next_state = state + action
        if next_state == 0: # 到达A左侧（终止）
            reward = 0
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            break
        elif next_state == n_states + 1: # 到达E右侧（终止）
            reward = 1
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            break
        else:
            reward = 0
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    # 记录价值函数
    V_history.append(V[1:n_states + 1].copy())

# 可视化
plt.figure(figsize=(10, 6))
# 绘制价值函数
for i in range(n_states):
    plt.plot([V_history[ep][i] for ep in range(n_episodes)], label=f'State {i+1}')
# 绘制真实价值函数
plt.plot(np.arange(n_episodes), [true_V[1]] * n_episodes, 'k--', label='True Values (A)')
plt.plot(np.arange(n_episodes), [true_V[2]] * n_episodes, 'k--')
plt.plot(np.arange(n_episodes), [true_V[3]] * n_episodes, 'k--')
plt.plot(np.arange(n_episodes), [true_V[4]] * n_episodes, 'k--')
plt.plot(np.arange(n_episodes), [true_V[5]] * n_episodes, 'k--')
plt.xlabel('Episodes')
plt.ylabel('Estimated Value')
plt.title('TD(0) Learning on Random Walk')
plt.legend()
plt.grid(True)
plt.show()

# 输出最终估计值与真实值对比
print("Final Estimated Values:", V[1:n_states + 1])
print("True Values:", true_V[1:n_states + 1])
