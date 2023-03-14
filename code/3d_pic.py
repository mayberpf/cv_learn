import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r) / r

# 创建绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(x, y, z, cmap='coolwarm')

# 绘制截面
z_section = z[50, :]  # 取y=0处的截面
x_section = x[50, :]
x_section = np.repeat(x_section.reshape(-1, 1), len(y), axis=1).T
y_section = np.repeat(y[50, :].reshape(1, -1), len(x), axis=0)
ax.plot_surface(x_section, y_section, np.ones_like(x_section)*0.3, cmap='coolwarm')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 去除坐标轴刻度和网格
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

# 添加图例
fig.colorbar(surf)

# 显示图形
plt.show()

