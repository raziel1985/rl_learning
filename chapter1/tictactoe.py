import numpy as np
import pickle
import os
from collections import defaultdict
import random

class TicTacToe:
    def __init__(self):
        # 初始化一个3x3的棋盘，0表示空位，1表示玩家1，-1表示玩家2
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 玩家1先手
        self.done = False
        self.winner = None
    
    def reset(self):
        # 重置游戏
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        # 返回当前棋盘状态的字符串表示
        return str(self.board.reshape(9))
    
    def get_available_actions(self):
        # 返回所有可用的动作（空位置）
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions
    
    def make_move(self, action):
        # 执行动作，返回新状态、奖励和是否结束
        i, j = action
        
        # 检查动作是否有效
        if self.board[i, j] != 0 or self.done:
            return self.get_state(), -10, True  # 无效动作，给予惩罚
        
        # 执行动作
        self.board[i, j] = self.current_player
        
        # 检查游戏是否结束
        if self._check_winner():
            self.done = True
            self.winner = self.current_player
            reward = 1.0  # 获胜奖励
        elif len(self.get_available_actions()) == 0:
            self.done = True
            self.winner = None
            reward = 0.5  # 平局奖励
        else:
            reward = 0.0  # 游戏继续
            # 切换玩家
            self.current_player = -self.current_player
        
        return self.get_state(), reward, self.done
    
    def _check_winner(self):
        # 检查行
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return True
        
        # 检查列
        for i in range(3):
            if abs(np.sum(self.board[:, i])) == 3:
                return True
        
        # 检查对角线
        if abs(np.sum(np.diag(self.board))) == 3:
            return True
        if abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        
        return False
    
    def render(self):
        # 打印棋盘
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print('-' * 13)
        for i in range(3):
            row = '| '
            for j in range(3):
                row += symbols[self.board[i, j]] + ' | '
            print(row)
            print('-' * 13)


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        # Q值表，默认为0
        # 二维字典：第一维表示状态（棋盘所有格子的状态），第二维表示动作（棋盘上的位置），值表示Q值
        # {
        #    "[ 0  1  0 -1  0  0  0  0  0]": {
        #       (0, 0): 0.75,
        #       (0, 2): 0.45,
        #       (1, 0): 0.2,
        #       ...
        #   },
        #   ...
        # },
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
    
    def get_action(self, state, available_actions):
        # 根据epsilon-greedy策略选择动作
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(available_actions)
        else:
            # 利用：选择Q值最大的动作
            q_values = [self.q_table[state][action] for action in available_actions]
            max_q = max(q_values)
            # 如果有多个最大Q值，随机选择一个
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, next_actions, done):
        # 更新Q值
        if done:
            # 如果游戏结束，没有下一个状态
            target = reward
        else:
            # 计算下一个状态的最大Q值
            next_q_values = [self.q_table[next_state][next_action] for next_action in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.gamma * max_next_q
        
        # 更新当前状态-动作对的Q值
        current_q = self.q_table[state][action]
        # Q-learning核型公式：Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
    
    def save_q_table(self, filename):
        # 保存Q表到文件
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, filename):
        # 从文件加载Q表
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                q_table = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), q_table)
                return True
        return False


def train(episodes=10000, save_interval=1000):
    env = TicTacToe()
    agent1 = QLearningAgent()  # 玩家1的代理
    agent2 = QLearningAgent()  # 玩家2的代理
    
    # 尝试加载已有的Q表
    model_path = "./chapter1/q_table_player1.pkl"
    if agent1.load_q_table(model_path):
        print("已加载玩家1的Q表")
    
    model_path = "./chapter1/q_table_player2.pkl"
    if agent2.load_q_table(model_path):
        print("已加载玩家2的Q表")
    
    # 训练循环
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        while not done:
            # 当前玩家选择动作
            available_actions = env.get_available_actions()
            if env.current_player == 1:
                action = agent1.get_action(state, available_actions)
            else:
                action = agent2.get_action(state, available_actions)
            
            # 执行动作
            next_state, reward, done = env.make_move(action)
            
            # 更新Q值
            next_available_actions = env.get_available_actions()
            if env.current_player == 1:  # 玩家2刚刚行动
                agent2.update(state, action, reward, next_state, next_available_actions, done)
            else:  # 玩家1刚刚行动
                agent1.update(state, action, reward, next_state, next_available_actions, done)
            
            state = next_state
        
        # 定期保存模型
        if episode % save_interval == 0:
            print(f"已完成 {episode}/{episodes} 轮训练")
            agent1.save_q_table("./chapter1/q_table_player1.pkl")
            agent2.save_q_table("./chapter1/q_table_player2.pkl")
    
    # 训练结束，保存最终模型
    agent1.save_q_table("./chapter1/q_table_player1.pkl")
    agent2.save_q_table("./chapter1/q_table_player2.pkl")
    print("训练完成！")


def play_against_ai():
    env = TicTacToe()
    agent = QLearningAgent(epsilon=0.0)  # 设置epsilon为0，使AI总是选择最优动作
    
    # 加载AI的Q表
    if not agent.load_q_table("./chapter1/q_table_player1.pkl"):
        print("未找到AI模型，请先训练模型")
        return
    
    # 决定谁先手
    human_player = int(input("你想要做先手(1)还是后手(-1)？ "))
    state = env.reset()
    done = False
    env.render()
    while not done:
        if env.current_player == human_player:
            # 人类玩家回合
            print("你的回合，请输入行和列 (1-3)：")
            try:
                i = int(input("行: "))
                j = int(input("列: "))
                action = (i - 1, j - 1)
                
                # 检查动作是否有效
                if action not in env.get_available_actions():
                    print("无效的动作，请重试")
                    continue
                
                state, reward, done = env.make_move(action)
            except ValueError:
                print("请输入有效的数字")
                continue
        else:
            # AI回合
            print("AI思考中...")
            available_actions = env.get_available_actions()
            action = agent.get_action(state, available_actions)
            state, reward, done = env.make_move(action)
            print(f"AI选择了位置: {action}")
        
        env.render()
        if done:
            if env.winner == human_player:
                print("恭喜，你赢了！")
            elif env.winner is None:
                print("平局！")
            else:
                print("AI赢了！")


if __name__ == "__main__":
    choice = input("请选择模式：1. 训练AI  2. 与AI对战  ")
    
    if choice == "1":
        episodes = int(input("请输入训练轮数（推荐10000轮以上）："))
        train(episodes=episodes)
    elif choice == "2":
        play_against_ai()
    else:
        print("无效的选择")

