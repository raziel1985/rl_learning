def iterative_policy_evalutation(policy, probability, reward, states, theta=1e-6, gamma=0.9):
    """
    策略评估
    :param policy: 策略
    :param probability: 状态转移概率
    :param reward: 即时奖励
    :param states: 所有状态
    :param theta: 阈值
    :param gamma: 折扣因子
    :return: 状态价值
    """
    V = {s : 0 for s in states} # 将所有状态的价值初始化为0
    iteration = 0
    while True:
        iteration += 1
        max_delta = 0
        print('Iteration %d' % iteration)
        for s in states:
            old_v = V[s]
            new_v = 0
            for a, action_prob in policy[s].items():
                for next_s, prob in probability[(s, a)]:
                    r = reward.get((s, a, next_s), 0)
                    new_v += action_prob * prob * (r + gamma * V[next_s])
            V[s] = new_v
            max_delta = max(max_delta, abs(old_v - new_v))
            print('State %s: %f' % (s, V[s]))
        print('Max delta: %f' % max_delta)
        if max_delta < theta:
            break
    return V    

policy = {
    's0': {'a0': 0.5, 'a1': 0.5},
    's1': {'a0': 0.5, 'a1': 0.5},
    's2': {'a0': 1.0}
}
probability = {
    ('s0', 'a0'): [('s0', 0.5), ('s1', 0.5)],
    ('s0', 'a1'): [('s1', 1.0)],
    ('s1', 'a0'): [('s0', 0.7), ('s1', 0.1), ('s2', 0.2)],
    ('s1', 'a1'): [('s0', 0.95), ('s1', 0.05)],
    ('s2', 'a0'): [('s2', 1.0)]
}
reward ={
    ('s0', 'a0', 's0'): 5,
    ('s0', 'a0', 's1'): -1,
    ('s0', 'a1', 's1'): -1,
    ('s1', 'a0', 's0'): -1,
    ('s1', 'a0', 's1'): -1,
    ('s1', 'a0', 's2'): 10,
    ('s1', 'a1', 's0'): 5,
    ('s1', 'a1', 's1'): -1,
    ('s2', 'a0', 's2'): 0
}
states = ['s0', 's1', 's2']

V = iterative_policy_evalutation(policy, probability, reward, states)
print("\n 最终结果")
for s in states:
    print('State %s: %f' % (s, V[s]))
