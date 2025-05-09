import numpy as np
import matplotlib.pyplot as plt

class GolfEnv:
    def __init__(self, size=20):
        self.size = size
        # 创建球场网格
        self.grid = np.zeros((size, size))
        # 设置洞的位置（中心位置）
        self.hole = (size//2, size//2)
        # 设置沙坑
        self.sand_traps = [(size//2-3, size//2-2), (size//2+2, size//2+3)]
        # 设置推杆最大距离
        self.putt_range = 3
        
        # 初始化状态价值函数，初始化为负无穷
        self.value_putt = np.full((size, size), float('-inf'))
        self.initialize_value_function()
    
    def initialize_value_function(self):
        """初始化状态价值函数"""
         # 洞的位置价值为0
        self.value_putt[self.hole] = 0
        changed = True
        while changed:
            changed = False
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) == self.hole or (i, j) in self.sand_traps:
                        # 洞和沙坑的价值永远不更新，前者为初始化的0，后者为初始化的-inf
                        continue
                    # 计算到洞的距离
                    dist = np.sqrt((i-self.hole[0])**2 + (j-self.hole[1])**2)
                    # 如果在推杆范围内
                    if dist <= self.putt_range:
                        new_value = -1  # 一杆进洞
                    else:
                        # 找到所有可以一杆打到的位置的最佳价值
                        best_value = float('-inf')
                        for x in range(max(0, i-self.putt_range), min(self.size, i+self.putt_range+1)):
                            for y in range(max(0, j-self.putt_range), min(self.size, j+self.putt_range+1)):
                                if np.sqrt((x-i)**2 + (y-j)**2) <= self.putt_range:
                                    best_value = max(best_value, self.value_putt[x, y])
                        if best_value != float('-inf'):
                            # 加上当前这一杆
                            new_value = best_value - 1
                        else:
                            # 如果没有可以一杆打到的位置
                            new_value = float('-inf')
                    if new_value != self.value_putt[i, j]:
                        self.value_putt[i, j] = new_value
                        changed = True
    

    def plot_value_function(self):
        """可视化状态价值函数"""
        plt.figure(figsize=(10, 8))
        
        # 创建掩码来隐藏无效值
        masked_value = np.ma.masked_where(self.value_putt == float('-inf'), self.value_putt)
        
        # 绘制价值函数
        plt.imshow(masked_value, cmap='viridis')
        plt.colorbar(label='Value')
        
        # 标记特殊位置
        plt.plot(self.hole[1], self.hole[0], 'r*', markersize=15, label='Hole')
        for trap in self.sand_traps:
            plt.plot(trap[1], trap[0], 'ys', markersize=10)
        
        # 绘制等值线
        contours = plt.contour(masked_value, levels=np.arange(-6, 1), colors='white', alpha=0.5)
        plt.clabel(contours, inline=True, fontsize=8)
        
        plt.title('Value Function for Putting Strategy')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.savefig('./chapter3/golf_value_function.png')
        plt.show()


if __name__ == "__main__":
    # 创建高尔夫环境并显示价值函数
    env = GolfEnv()
    env.plot_value_function()
