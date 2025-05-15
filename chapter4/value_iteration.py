import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 定义动作
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 左, 下, 右, 上
action_symbols = ['←', '↓', '→', '↑']

def env_step(state, action):
    """
    环境的一步
    :param state: 状态
    :param action: 动作
    :return: 奖励, 下一个状态
    """
    # 检查状态是否在网格内
    assert state not in [(0, 0), (3, 3)], "Terminal states cannot be chosen"
    next_row = state[0] + action[0]
    next_col = state[1] + action[1]
    if next_row < 0 or next_row >= 4 or next_col < 0 or next_col >= 4:
        # 出界, 回到原来的状态
        return -1, state
    else:
        # 没有出界, 进入下一个状态
        return -1, (next_row, next_col)


def value_iteration(grid_world, actions, gamma, theta, termal_states):
    """
    价值迭代
    :param grid_world: 网格世界
    :param actions: 动作
    :param gamma: 折扣因子
    :param theta: 阈值
    :param termal_states: 终止状态
    :return: 新的价值函数
    """
    while True:
        delta = 0
        for row in range(grid_world.shape[0]):
            for col in range(grid_world.shape[1]):
                if (row, col) in termal_states:
                    continue
                old_value = grid_world[row, col]
                new_value = float('-inf')
                for action in actions:
                    reward, next_state = env_step((row, col), action)
                    next_value = grid_world[next_state]
                    new_value = max(new_value, reward + gamma * next_value)
                grid_world[row, col] = new_value
                delta = max(delta, np.abs(old_value - new_value))
        if delta < theta:
            break
    return grid_world


def get_optimal_policy(grid_world, actions, gamma, termal_states):
    """
    获取最优策略
    :param grid_world: 网格世界
    :param actions: 动作
    :param gamma: 折扣因子
    :param termal_states: 终止状态
    :return: 最优策略
    """
    optimal_policy = []
    for row in range(grid_world.shape[0]):
        policy_row = []
        for col in range(grid_world.shape[1]):
            if (row, col) in termal_states:
                policy_row.append('')
                continue
            q_values = []
            for i, action in enumerate(actions):
                reward, next_state = env_step((row, col), action)
                q_values.append(reward + gamma * grid_world[next_state])
            best_actions = np.where(np.array(q_values) == max(q_values))[0]
            policy_row.append("".join([action_symbols[i] for i in best_actions]))
        optimal_policy.append(policy_row)
    return optimal_policy


def plot_gridworld(data, annots, curr_row):
    """
    绘制网格世界
    :param data: 数据
    :param annots: 标注
    :param curr_row: 当前行
    :return: None
    """
    # 创建子图
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"价值函数 (k={curr_row})", f"最优策略 (k={curr_row})"),
                        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]])

    # 创建 'Blues' 色彩方案
    blues = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']

    # 价值函数热图
    fig.add_trace(
        go.Heatmap(z=data,
                   text=[[f'{val:.1f}' for val in row] for row in data],
                   texttemplate="%{text}",
                   textfont={"size": 16, "color": "black"},
                   colorscale=blues,
                   showscale=False),
        row=1, col=1
    )

    # 策略热图
    fig.add_trace(
        go.Heatmap(z=data,
                   text=annots,
                   texttemplate="%{text}",
                   textfont={"size": 20, "color": "black"},
                   colorscale=blues,
                   showscale=False),
        row=1, col=2
    )

    fig.update_layout(
        title=f'最优价值函数和策略 (k={"∞" if curr_row == -1 else curr_row})',
        title_font=dict(size=16, color='black', family="Arial, sans-serif"),
        title_x=0.5,
        width=900,
        height=400,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    for i in range(1, 3):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=i, autorange='reversed')
    fig.show()


grid_world = np.zeros(shape=(4, 4))
gamma = 1.0
theta = 0.01
terminal_states = [(0, 0), (3, 3)]

iterations_to_plot = [0, 1, 2, 3, 10]
for i in range(1000):
    if i in iterations_to_plot:
        optimal_policy = get_optimal_policy(grid_world, actions, gamma, terminal_states)
        plot_gridworld(grid_world, optimal_policy, i)
    old_grid_world = grid_world.copy()
    grid_world = value_iteration(grid_world, actions, gamma, theta, terminal_states)
    if np.max(np.abs(grid_world - old_grid_world)) < theta:
        print(f'在迭代 {i + 1} 时收敛')
        optimal_policy = get_optimal_policy(grid_world, actions, gamma, terminal_states)
        plot_gridworld(grid_world, optimal_policy, i + 1)
        break
