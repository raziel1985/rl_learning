import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_actions(stake):
    """
    获取所有可能的动作
    :param stake: 赌注
    :return: 所有可能的动作
    """
    max_stake = min(stake, 100 - stake)
    # 忽略动作0
    return np.arange(1, max_stake + 1)


def get_next_state(V, state, actions):
    """
    获取下一个状态
    :param V: 价值函数
    :param state: 当前状态
    :param actions: 所有可能的动作
    :return: 下一个状态
    """
    wins = V[state + actions]
    fails = V[state - actions]
    return (wins, fails)


def value_iteration(prob_h=0.4, theta=1e-5):
    global history
    global full_policy
    # 初始化价值函数, V(100) = 1, 其他为0
    V = np.zeros(shape=(101,))
    V[100] = 1
    policy = np.zeros(shape=(101,))

    while True:
        old_V = np.copy(V)
        for state in range(1, 100):
            actions = get_actions(state)
            wins, fails = get_next_state(V, state, actions)
            values = prob_h * wins + (1 - prob_h) * fails
            # 选择最大价值的所有动作
            best_actions = np.where(values == np.max(values))[0]
            best_action = best_actions[0]
            V[state] = values[best_action]
            policy[state] = actions[best_action]
            full_policy[state] = actions[best_actions]
        history.append(np.copy(V))
        if np.max(np.abs(V - old_V)) < theta:
            break
    print("总扫描次数: ", len(history))
    return V, policy


# 绘制价值估计图
def plot_value_estimates(history, sweeps):
    fig = go.Figure()
    colors = ['cornflowerblue', 'tomato', 'lightseagreen', 'indigo']

    annotations = []

    for i, sweep in enumerate(sweeps):
        x = np.arange(1, 100)
        y = history[sweep][1:100]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'扫描 {sweep + 1}',
            line=dict(color=colors[i])
        ))

        # 为每次扫描选择不同的注释位置
        if i < 3 or i == len(sweeps) - 1:
            if i == 0:
                annotation_x, annotation_y = 25, y[24]
            elif i == 1:
                annotation_x, annotation_y = 50, y[49]
            elif i == 2:
                annotation_x, annotation_y = 75, y[74]
            else:  # 最后一次扫描
                annotation_x, annotation_y = 85, y[84]

            annotations.append(dict(
                x=annotation_x, y=annotation_y,
                text=f"扫描 {sweep + 1}" if i < 3 else "最终价值函数",
                showarrow=True, arrowhead=2,
                ax=-40 if i < 3 else 40,
                ay=-40 if i < 3 else 20,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            ))

    fig.update_layout(
        title='价值估计',
        xaxis_title='资本',
        yaxis_title='价值估计',
        legend_title='扫描次数',
        font=dict(size=12),
        annotations=annotations,
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
    )
    fig.show()


# 绘制最终策略图
def plot_final_policy(policy):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(1, 100),
        y=policy[1:100],
        mode='lines',
        line=dict(shape='hv')
    ))

    fig.update_layout(
        title='最终策略',
        xaxis_title='资本',
        yaxis_title='最终策略',
        font=dict(size=12)
    )
    fig.show()


# 绘制完整策略图
def plot_full_policy(full_policy):
    fig = go.Figure()

    for key, values in full_policy.items():
        keys = [key] * len(values)
        fig.add_trace(go.Scatter(
            x=keys,
            y=values,
            mode='markers',
            marker=dict(size=4, color='royalblue')
        ))

    fig.update_layout(
        title='完整策略',
        xaxis_title='资本',
        yaxis_title='完整策略',
        font=dict(size=12)
    )
    fig.show()


history = []
full_policy = dict()

prob_head = 0.4
theta = 1e-6
V, pi = value_iteration(prob_h=prob_head, theta=theta)

sweeps = [0, 1, 2, len(history) -1]
plot_value_estimates(history, sweeps)

# 绘制最终策略
plot_final_policy(pi)

# 绘制完整策略
plot_full_policy(full_policy)
