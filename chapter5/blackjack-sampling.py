import numpy as np
import gymnasium as gym
import plotly.graph_objects as go
from typing import Any
from tqdm import tqdm

# 目标策略: 大于等于20点停止，否则继续要牌
def prob_target_policy(player_sum:int, action: int) -> float:
    if player_sum >= 20: 
        # 大于等于20点停止
        return 1.0 if action == 0 else 0.0
    else: 
        # 小于20点继续要牌
        return 1.0 if action == 1 else 0.0

# 行为策略：随机选择动作
def prob_behavior_policy() -> float:
    return 0.5

# 随机选择动作
def get_action() -> int:
    return np.random.choice([0, 1])

# 计算均方误差
def mse(val_1: float, val_2: float) -> float:
    return np.sqrt((val_1 - val_2) ** 2)

# 初始化状态, 直到状态等于start_state
def init_start_state(env: Any, start_state: tuple) -> tuple:
    obvervation, _ = env.reset()
    while tuple(obvervation) != start_state:
        obvervation, _ = env.reset()
    return obvervation

def monte_carlo_importance_sampling(env: Any, start_state: tuple, target_val: float, gamma: float, total_rounds: int, episodes_per_round:int) -> dict:
    value_hist = {
        'ordinary': np.zeros((total_rounds, episodes_per_round)),
        'weighted': np.zeros((total_rounds, episodes_per_round))
    }
    for r in range(total_rounds):
        print(f'正在进行第{r+1}轮采样')
        V_ord, V_wei, rho, count = 0, 0, 0, 0
        for t in tqdm(range(episodes_per_round)):
            state = init_start_state(env, start_state)
            action = get_action()
            episode = []
            while True:
                next_state, reward, terminated, _, _ = env.step(action)
                episode.append((tuple(state), action, reward))
                if terminated:
                    break
                state = next_state
                action = get_action()

            # 计算重要性采样的权重
            G, W = 0, 1.0
            for state, action, reward in reversed(episode):
                G = gamma * G + reward
                W *= prob_target_policy(state[0], action) / prob_behavior_policy()
                if state == start_state:
                    V_ord += W * G
                    count += 1
                    V_wei += W * G
                    rho += W
            value_hist['weighted'][r, t] = mse(V_wei / rho if rho != 0 else 0, target_val)
            value_hist['ordinary'][r, t] = mse(V_ord / count, target_val)
        plot_result(value_hist, r + 1)
    return value_hist

# 可视化结果
def plot_result(value_hist: dict, round: int) -> None:
    ord_hist = value_hist['ordinary'].mean(axis=0)
    weighted_hist = value_hist['weighted'].mean(axis=0)
    fig = go.Figure()
    x = np.arange(1, len(ord_hist) + 1)
    fig.add_trace(go.Scatter(x=x, y=weighted_hist, mode='lines', name='加权重要性采样',
                             line=dict(color='tomato', width=2)))
    fig.add_trace(go.Scatter(x=x, y=ord_hist, mode='lines', name='普通重要性采样',
                             line=dict(color='lightseagreen', width=2)))
    fig.update_layout(
        title='21点游戏 - 蒙特卡洛重要性采样',
        xaxis_title='回合数 (对数刻度)',
        yaxis_title=f'均方误差 ({round}次运行的平均值)',
        xaxis_type='log',
        xaxis=dict(tickmode='array', tickvals=[1, 10, 100, 1000, 10000],
                   ticktext=['1', '10', '100', '1000', '10,000']),
        yaxis=dict(range=[0, 2]),
        legend=dict(x=0.7, y=0.98),
        width=800,
        height=500
    )
    fig.show()


env = gym.make('Blackjack-v1', sab=True)
start_state = (13, 2, 1) # 初始状态, 玩家的点数为13, 庄家的点数为2, 玩家没有 Ace
target_val = -0.27726 # 目标状态的价值
gamma = 1.0
total_rounds = 20
episodes_per_round = 1000
value_hist = monte_carlo_importance_sampling(env, start_state, target_val, gamma, total_rounds, episodes_per_round)
