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
    

def iterative_policy_evalutation(grid_world, actions, probs, gamma, termal_states):
    """
    迭代策略评估
    :param grid_world: 网格世界（价值函数）
    :param actions: 动作
    :param probs: 动作概率
    :param gamma: 折扣因子
    :param termal_states: 终止状态
    :return: 新的价值函数, 最大差值
    """
    Q = np.zeros(shape=(grid_world.shape[0], grid_world.shape[1], len(actions)))
    for row in range(grid_world.shape[0]):
        for col in range(grid_world.shape[1]):
            if (row, col) in termal_states:
                continue
            for i, action in enumerate(actions):
                reward, next_state = env_step((row, col), action)
                next_value = grid_world[next_state]
                Q[row, col, i] = probs[i] * (reward + gamma * next_value)
    new_grid_world = Q.sum(axis=-1)
    max_diff = np.abs(new_grid_world - grid_world).max()
    return new_grid_world, max_diff


def greed_policy(grid_world, actions, probs, gamma, termal_states):
    """
    贪婪策略
    :param grid_world: 网格世界
    :param actions: 动作
    :param probs: 动作概率
    :param gamma: 折扣因子
    :param termal_states: 终止状态
    :return: 贪婪策略
    """
    Q = np.zeros(shape=(grid_world.shape[0], grid_world.shape[1], len(actions)))
    greeed_acts = []
    for row in range(grid_world.shape[0]):
        acts_row = []
        for col in range(grid_world.shape[1]):
            if (row, col) in termal_states:
                acts_row.append('')
                continue
            for i, action in enumerate(actions):
                reward, next_state = env_step((row, col), action)
                next_value = grid_world[next_state]
                Q[row, col, i] = probs[i] * (reward + gamma * next_value)
            greedy_sels = np.where(np.abs(Q[row, col] - Q[row, col].max()) < 0.0001)[0]
            acts = "".join([action_symbols[i] for i in greedy_sels])
            acts_row.append(acts)
        greeed_acts.append(acts_row)
    return greeed_acts


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

# 动作概率, 均匀分布
probs = [1 / len(actions)] * len(actions)
termal_states = [(0, 0), (3, 3)]
iterations_to_plot = [0, 1, 2, 3, 10]
# 迭代策略评估
for i in range(1000):
    if i in iterations_to_plot:
        # 绘制网格世界
        plot_gridworld(grid_world, greed_policy(grid_world, actions, probs, gamma, termal_states), i)
    grid_world, delta = iterative_policy_evalutation(grid_world, actions, probs, gamma, termal_states)
    if delta < theta:
        # 收敛, 绘制网格世界
        print(f'迭代次数: {i + 1} 时收敛')
        plot_gridworld(grid_world, greed_policy(grid_world, actions, probs, gamma, termal_states), i + 1)
        break
