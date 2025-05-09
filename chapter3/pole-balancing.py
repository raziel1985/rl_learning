import gymnasium as gym

# 创建环境
env = gym.make('CartPole-v1', render_mode="human")

n_episodes = 20
for episode in range(n_episodes):
    state, info = env.reset()
    total_reward = 0
    done = False
    truncated = False
    while not done:
        # 渲染环境
        env.render()
        # 随机选择动作
        action = env.action_space.sample()
        # 执行动作
        state, reward, done, truncated, info = env.step(action)
        # 更新总奖励
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# 关闭环境
env.close()
