import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 定义动作
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 左, 下, 右, 上
action_symbols = ['←', '↓', '→', '↑']

# 定义状态转移函数
def get_reward(state:tuple, action:tuple):
    """
    定义状态转移函数
    :param state: 状态 (row, col)
    :param action: 动作 (row, col)
    :return: 奖励, 下一个状态(row, col)
    """
    if state == (0, 1):
        # 特殊状态, 奖励 10 并转移到状态 (4, 1)
        return 10, (4, 1)
    if state == (0, 3):
        # 特殊状态, 奖励 5 并转移到状态 (2, 3)
        return 5, (2, 3)
    
    # 普通状态转移
    next_row = state[0] + action[0]
    next_col = state[1] + action[1]
    # 检查边界, 如果越界则不转移, 奖励 -1
    if not (0 <= next_row < 10 and 0 <= next_col < 10):
        return -1, state
    # 否则转移到下一个状态, 奖励 0
    return 0, (next_row, next_col)


# 定义状态价值函数
def grid_state_value_func(grid_world: np.ndarray, actions: list, gamma: float):
    """
    计算状态价值函数
    :param grid_world: 网格世界
    :param actions: 动作列表
    :param gamma: 折扣因子
    :return: 新的网格世界, 最优动作
    """
    optim_acts = []
    new_grid = np.copy(grid_world)
    for row in range(10):
        for col in range(10):
            value_candidates = np.zeros(len(actions))
            for i, act in enumerate(actions):
                reward, next_state = get_reward((row, col), act)
                next_value = grid_world[next_state]
                # 记录每个动作的价值
                value_candidates[i] = reward + gamma * next_value
            # 记录最优动作
            new_grid[row, col] = value_candidates.max()
            max_args = np.where(value_candidates == value_candidates.max())[0]
            select_act = ''.join([action_symbols[i] for i in max_args])
            optim_acts.append(select_act)
    # 返回新的网格世界, 最优动作
    return new_grid, np.array(optim_acts).reshape(10, 10)


# 可视化网格世界
def plot_grid(grid_world: np.ndarray, optim_acts: list, iteration: int = 0):
    """
    可视化网格世界
    :param grid_world: 网格世界
    :param optim_acts: 最优动作
    :param iteration: 迭代次数
    :return: None
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("最优价值函数", "最优策略"),
                        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]])

    # 价值函数热图
    fig.add_trace(
        go.Heatmap(z=grid_world, text=[[f'{val:.1f}' for val in row] for row in grid_world],
                   texttemplate="%{text}", textfont={"size": 16, "color": "white"},
                   colorscale='Viridis', showscale=True,
                   colorbar=dict(title='价值', ticks='outside')),
        row=1, col=1
    )

    # 策略热图
    fig.add_trace(
        go.Heatmap(z=grid_world, text=optim_acts, texttemplate="%{text}",
                   textfont={"size": 16, "color": "black"},
                   colorscale='Viridis', showscale=False),
        row=1, col=2
    )

    fig.update_layout(
        title=f'最优价值函数和策略 (迭代 {iteration})',
        title_font_size=16, title_x=0.5,
        width=1800, height=900,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    for i in range(1, 3):
        fig.update_xaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='LightGrey', row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='LightGrey', row=1, col=i, autorange='reversed')
    fig.show()


# 初始化网格世界
grid_world = np.zeros((10, 10))
gamma = 0.9

# 迭代计算状态价值函数
for i in range(100):
    pre_grid = np.copy(grid_world)
    grid_world, optim_acts = grid_state_value_func(grid_world, actions, gamma)
    if np.allclose(pre_grid, grid_world, rtol=0.01):
        # 收敛输出结果
        print(f'Converged at iteration {i}')
        print(grid_world)
        print(optim_acts)
        plot_grid(grid_world, optim_acts, i)
        break
