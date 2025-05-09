import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

gamma = 0.9 # 折扣因子
alpha = 0.5 # 高电量搜索成功后仍是高电量的概率， 1-alpha 为低电量的概率
beta = 0.5  # 低电量搜索成功后仍是低电量的概率, 1-beta 为空电的概率
rs = 2  # 搜索奖励
rw = 1  # 等待奖励
epsilon = 1e-6

def value_iteration():
    """
    价值迭代
    :return: 高电量和低电量的价值函数
    """
    v_h = 0
    v_l = 0
    while True:
        v_h_new = max(
            rs + gamma * (alpha * v_h + (1 - alpha) * v_l), # search
            rw + gamma * v_h # wait
        )
        v_l_new = max(
            rs + gamma * (beta * v_l + (1 - beta) * (v_h - 3)), # search
            rw + gamma * v_l, # wait
            gamma + v_h # recharge
        )
        if abs(v_h_new - v_h) < epsilon and abs(v_l_new - v_l) < epsilon:
            break
        v_h, v_l = v_h_new, v_l_new
    return v_h, v_l


def optimal_policy(v_h, v_l):
    """
    最优策略
    :param v_h: 高电量的价值函数
    :param v_l: 低电量的价值函数
    :return: 高电量和低电量的最优策略
    """
    policy_h = "search" if rs + gamma * (alpha * v_h + (1 - alpha) * v_l) > rw + gamma * v_h else "wait"
    search_value = rs + gamma * (beta * v_l + (1 - beta) * (v_h - 3))
    waite_value = rw + gamma * v_l
    recharge_value = gamma + v_h
    if search_value > waite_value and search_value > recharge_value:
        policy_l = "search"
    elif waite_value > recharge_value:
        policy_l = "wait"
    else:
        policy_l = "recharge"
    return policy_h, policy_l

v_h, v_l = value_iteration()
policy_h, policy_l = optimal_policy(v_h, v_l)
print(f"高电量的价值函数: {v_h:.2f}")
print(f"低电量的价值函数: {v_l:.2f}")
print(f"高电量的最优策略: {policy_h}")
print(f"低电量的最优策略: {policy_l}")

# 创建表格数据
table_data = [
   ['high', 'seach', 'hight', str(alpha), str(rs)],
   ['high', 'search', 'low', str(1-alpha), str(rs)],
   ['low', 'search', 'high', str(1-beta), str(-3)],
   ['low', 'search', 'low', str(beta), str(rs)],
   ['high', 'wait', 'high', '1', str(rw)],
   ['high', 'wait', 'low', '0', '-'],
   ['low',' wait', 'high', '0', '-'],
   ['low', 'wait', 'low', '1', str(rw)],
   ['low', 'recharge', 'high', '1', '0'],
   ['low','recharge', 'low', '0', '-']
]

# 使用Plotly创建表格
fig = go.Figure(data=[go.Table(
    header=dict(values=['state', 'action', 'next state', 'transition probability', 'reward'], fill_color='paleturquoise', align='left'),
    cells=dict(values=list(zip(*table_data)), fill_color='lavender', align='left'))
])
fig.update_layout(title='回收机器人问题状态转移和奖励表')
fig.show()

# 创建最优价值函数和策略的可视化
states = ["High", "Low"]
values = [v_h, v_l]
policies = [policy_h, policy_l]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Value Function", "Policy"))
fig.add_trace(go.Bar(x=states, y=values, text=[f'{v:.2f}' for v in values], textposition='auto'), row=1, col=1)
fig.add_trace(go.Bar(x=states, y=[1, 1], text=policies, textposition='auto'), row=1, col=2)
fig.update_layout(title='最优价值函数和策略', showlegend=False, height=500, width=800)
fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Policy", row=1, col=2)
fig.show() 
