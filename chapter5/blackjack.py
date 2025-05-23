import numpy as np
import gymnasium as gym
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 创建21点游戏环境
# Jack, Queen, King 都计作10, Ace 可以计作1或11, 数字牌2~10计作其数字值
# 初始：玩家和庄家个获得2张牌，玩家全明，庄家一明一暗
# 玩家行动：玩家可以选择抽牌或停止抽牌
# 庄家行动：玩家停止抽牌后, 庄家会翻开暗牌, 然后庄家会继续抽牌直到其牌面总和大于等于17
# 游戏结束：如果玩家的牌面总和超过21, 则玩家输; 如果庄家的牌面总和超过21, 则玩家赢; 否则，比较玩家和庄家的牌面总和, 较大者赢
env = gym.make("Blackjack-v1", natural=False, sab=False)

# 定义策略
def policy(player_currr_sum: int) -> int:
    """
    定义一个简单的策略: 如果玩家的当前牌面总和小于20, 则继续抽牌, 否则停止抽牌。
    """
    if player_currr_sum < 20:
        return 1  # 抽牌
    else:
        return 0  # 停止抽牌


# 蒙特卡洛预测算法
def monte_carlo_prediction(num_episodes, gamma=1.0):
    V = np.zeros(shape=(22, 11, 2)) # 初始化价值函数 (玩家当前牌面总和, 庄家第一张明牌, 玩家是否有Ace)
    returns = np.zeros_like(V, dtype=np.float32) # 记录每个状态的累计回报
    counts = np.zeros_like(V, dtype=np.int16) # 记录每个状态被访问的次数

    for _ in range(num_episodes):
        observation, _ = env.reset() # observation: (玩家当前牌面总和, 庄家第一张明牌, 是否有Ace)
        episode = []
        while True:
            action = policy(observation[0])
            next_observation, reward, done, _, _ = env.step(action) # reward: 1代表玩家赢, -1代表玩家输, 0代表游戏继续; done: True代表游戏结束
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_observation

        # 计算每个状态的累计回报
        G = 0
        seen = set()
        for observation, action, reward in reversed(episode):
            G = gamma * G + reward
            if observation not in seen:
                seen.add(observation)
                returns[observation] += G
                counts[observation] += 1
                V[observation] = returns[observation] / counts[observation]
    return V


# 可视化函数
def plot_value_function(V, title):
    x = np.arange(1, 11) # 庄家明牌为1-10
    y = np.arange(12, 22) # 玩家总和为12-21
    X, Y = np.meshgrid(x, y)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('有可用A', '无可用A'),
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    for i, ace in enumerate([1, 0]):
        Z = V[12:22, 1:11, ace] # 玩家总和为12-21, 庄家明牌为1-10
        fig.add_trace(
            go.Surface(z=Z, x=X, y=Y, colorscale='Viridis',
                       colorbar=dict(title='价值', len=0.5, y=0.5-0.25*i, thickness=10)),
            row=1, col=i+1
        )
    fig.update_layout(title_text=title, autosize=False,
                      width=2000, height=1000,
                      scene = dict(
                          xaxis_title='庄家明牌',
                          yaxis_title='玩家总和',
                          zaxis_title='价值'),
                      scene2 = dict(
                          xaxis_title='庄家明牌',
                          yaxis_title='玩家总和',
                          zaxis_title='价值'))
    fig.show()


# 运行蒙特卡洛预测算法
num_episodes = 100000
V = monte_carlo_prediction(num_episodes)

# 可视化结果
plot_value_function(V, f'21点游戏状态价值函数 (迭代 {num_episodes} 次)')
