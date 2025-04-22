import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

class Bandit:
    """
    多臂老虎机环境
    """
    def __init__(self, k: int, true_means: List[float] = None):
        """
        初始化多臂老虎机环境
        
        参数:
            k: 拉杆数量
            true_means: 每个拉杆的真实期望奖励，如果为None则随机生成
        """
        self.k = k
        
        # 如果没有提供真实期望奖励，则随机生成
        if true_means is None:
            self.true_means = np.random.normal(0, 1, k)
        else:
            self.true_means = np.array(true_means)
            
        # 最优拉杆的索引
        self.optimal_arm = np.argmax(self.true_means)
    
    def pull(self, arm: int) -> float:
        """
        拉动指定的拉杆，返回奖励
        
        参数:
            arm: 要拉动的拉杆索引
            
        返回:
            奖励值
        """
        # 奖励服从以真实期望为中心的正态分布
        return np.random.normal(self.true_means[arm], 1)


class EpsilonGreedy:
    """
    ε-贪婪算法
    """
    def __init__(self, k: int, epsilon: float = 0.1):
        """
        初始化ε-贪婪算法
        
        参数:
            k: 拉杆数量
            epsilon: 探索概率
        """
        self.k = k
        self.epsilon = epsilon
        self.q_values = np.zeros(k)  # 每个拉杆的估计价值
        self.action_counts = np.zeros(k)  # 每个拉杆被选择的次数
    
    def select_action(self) -> int:
        """
        选择一个拉杆
        
        返回:
            选择的拉杆索引
        """
        # 以epsilon的概率进行探索
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        # 以1-epsilon的概率选择当前估计价值最高的拉杆
        else:
            return np.argmax(self.q_values)
    
    def update(self, action: int, reward: float) -> None:
        """
        更新估计价值
        
        参数:
            action: 选择的拉杆索引
            reward: 获得的奖励
        """
        self.action_counts[action] += 1
        # 增量更新公式
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]


class UCB:
    """
    上置信界(UCB)算法
    """
    def __init__(self, k: int, c: float = 2.0):
        """
        初始化UCB算法
        
        参数:
            k: 拉杆数量
            c: 探索参数
        """
        self.k = k
        self.c = c
        self.q_values = np.zeros(k)  # 每个拉杆的估计价值
        self.action_counts = np.zeros(k)  # 每个拉杆被选择的次数
        self.t = 0  # 总步数
    
    def select_action(self) -> int:
        """
        选择一个拉杆
        
        返回:
            选择的拉杆索引
        """
        # 确保每个拉杆至少被选择一次
        for arm in range(self.k):
            if self.action_counts[arm] == 0:
                return arm
        
        # 计算UCB值
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float) -> None:
        """
        更新估计价值
        
        参数:
            action: 选择的拉杆索引
            reward: 获得的奖励
        """
        self.t += 1
        self.action_counts[action] += 1
        # 增量更新公式
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]


class ThompsonSampling:
    """
    汤普森采样算法
    """
    def __init__(self, k: int):
        """
        初始化汤普森采样算法
        
        参数:
            k: 拉杆数量
        """
        self.k = k
        # 每个拉杆的Beta分布参数
        self.alpha = np.ones(k)
        self.beta = np.ones(k)
    
    def select_action(self) -> int:
        """
        选择一个拉杆
        
        返回:
            选择的拉杆索引
        """
        # 从每个拉杆的Beta分布中采样
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.k)]
        return np.argmax(samples)
    
    def update(self, action: int, reward: float) -> None:
        """
        更新Beta分布参数
        
        参数:
            action: 选择的拉杆索引
            reward: 获得的奖励
        """
        # 这里假设奖励是0或1，如果不是，需要进行转换
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1


def run_experiment(bandit: Bandit, agent, steps: int) -> Tuple[List[float], List[float]]:
    """
    运行实验
    
    参数:
        bandit: 多臂老虎机环境
        agent: 算法代理
        steps: 实验步数
        
    返回:
        rewards: 每一步的奖励
        optimal_actions: 每一步是否选择了最优拉杆
    """
    rewards = []
    optimal_actions = []
    
    for _ in range(steps):
        # 选择拉杆
        action = agent.select_action()
        
        # 拉动拉杆获取奖励
        reward = bandit.pull(action)
        
        # 更新代理
        agent.update(action, reward)
        
        # 记录结果
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.optimal_arm else 0)
    
    return rewards, optimal_actions


def plot_results(rewards_list: List[List[float]], optimal_actions_list: List[List[float]], labels: List[str]):
    """
    绘制实验结果
    
    参数:
        rewards_list: 每个算法的奖励列表
        optimal_actions_list: 每个算法选择最优拉杆的比例列表
        labels: 算法标签
    """
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS上通常可用的支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.figure(figsize=(15, 6))
    
    # 绘制平均奖励
    plt.subplot(1, 2, 1)
    for i, rewards in enumerate(rewards_list):
        # 计算累积平均奖励
        cumulative_rewards = np.cumsum(rewards)
        steps = np.arange(1, len(rewards) + 1)
        average_rewards = cumulative_rewards / steps
        plt.plot(average_rewards, label=labels[i])
    
    plt.xlabel('步数')
    plt.ylabel('平均奖励')
    plt.title('平均奖励随时间变化')
    plt.legend()
    
    # 绘制选择最优拉杆的比例
    plt.subplot(1, 2, 2)
    for i, optimal_actions in enumerate(optimal_actions_list):
        # 计算选择最优拉杆的累积比例
        cumulative_optimal = np.cumsum(optimal_actions)
        steps = np.arange(1, len(optimal_actions) + 1)
        optimal_percentages = cumulative_optimal / steps
        plt.plot(optimal_percentages, label=labels[i])
    
    plt.xlabel('步数')
    plt.ylabel('选择最优拉杆的比例')
    plt.title('选择最优拉杆的比例随时间变化')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/roger/Dev/rl_learning/chapter2/bandit_results.png')
    plt.show()


if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    np.random.seed(42)
    random.seed(42)
    
    # 实验参数
    k = 10  # 拉杆数量
    steps = 1000  # 实验步数
    runs = 100  # 实验重复次数
    
    # 创建多臂老虎机环境
    true_means = np.random.normal(0, 1, k)
    print(f"真实期望奖励: {true_means}")
    print(f"最优拉杆: {np.argmax(true_means)}")
    
    # 算法列表
    algorithms = [
        (EpsilonGreedy(k, epsilon=0.1), "ε-贪婪 (ε=0.1)"),
        (EpsilonGreedy(k, epsilon=0.01), "ε-贪婪 (ε=0.01)"),
        (UCB(k, c=2.0), "UCB (c=2.0)"),
        (ThompsonSampling(k), "汤普森采样")
    ]
    
    # 存储每个算法的结果
    all_rewards = [[] for _ in range(len(algorithms))]
    all_optimal_actions = [[] for _ in range(len(algorithms))]
    
    # 运行多次实验并取平均
    for run in range(runs):
        bandit = Bandit(k, true_means)
        
        for i, (agent, _) in enumerate(algorithms):
            # 重置代理
            if isinstance(agent, EpsilonGreedy):
                agent.q_values = np.zeros(k)
                agent.action_counts = np.zeros(k)
            elif isinstance(agent, UCB):
                agent.q_values = np.zeros(k)
                agent.action_counts = np.zeros(k)
                agent.t = 0
            elif isinstance(agent, ThompsonSampling):
                agent.alpha = np.ones(k)
                agent.beta = np.ones(k)
            
            # 运行实验
            rewards, optimal_actions = run_experiment(bandit, agent, steps)
            
            # 存储结果
            if run == 0:
                all_rewards[i] = rewards
                all_optimal_actions[i] = optimal_actions
            else:
                all_rewards[i] = [a + b for a, b in zip(all_rewards[i], rewards)]
                all_optimal_actions[i] = [a + b for a, b in zip(all_optimal_actions[i], optimal_actions)]
    
    # 计算平均结果
    for i in range(len(algorithms)):
        all_rewards[i] = [r / runs for r in all_rewards[i]]
        all_optimal_actions[i] = [a / runs for a in all_optimal_actions[i]]
    
    # 绘制结果
    labels = [label for _, label in algorithms]
    plot_results(all_rewards, all_optimal_actions, labels)
