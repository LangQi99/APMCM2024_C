import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 读取数据函数


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        # 第一行包含年份
        years = np.array([int(x) for x in lines[0].split()[1:]])
        # 读取其他数据行
        data = {}
        for line in lines[1:]:
            parts = line.split()
            data[parts[0]] = np.array([float(x) for x in parts[1:]])
    return years, data


# 读取所有数据文件
years, data1 = read_data('data_1.txt')
_, data2 = read_data('data_2.txt')
_, data3 = read_data('data_3.txt')

# 准备数据
plots = [
    ('GDP-Year-Pets', years, data1['cat'], data1['dog'], data2['GDP']),
    ('Pet Economy-Year-Pets', years,
     data1['cat'], data1['dog'], data3['pet_industry_economy'])
]

# 分别创建两个图形
for i, (title, x, cat, dog, y) in enumerate(plots):
    # 创建新的图形窗口
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图和回归平面
    ax.scatter(x, y, cat, color='red', label='Cat')
    ax.scatter(x, y, dog, color='blue', label='Dog')

    # 准备回归数据（分别为猫和狗）
    X_cat = np.column_stack((x, y))
    X_dog = np.column_stack((x, y))

    regressor_cat = LinearRegression()
    regressor_dog = LinearRegression()

    regressor_cat.fit(X_cat, cat)  # 修改目标变量
    regressor_dog.fit(X_dog, dog)  # 修改目标变量

    # 生成网格数据
    x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 30),
                                 np.linspace(y.min(), y.max(), 30))

    # 预测值（分别为猫和狗）
    z_pred_cat = regressor_cat.predict(np.column_stack(
        (x_grid.ravel(), y_grid.ravel()))).reshape(x_grid.shape)
    z_pred_dog = regressor_dog.predict(np.column_stack(
        (x_grid.ravel(), y_grid.ravel()))).reshape(x_grid.shape)

    # 绘制预测平面
    surf_cat = ax.plot_surface(
        x_grid, y_grid, z_pred_cat, alpha=0.3, cmap='Reds')
    surf_dog = ax.plot_surface(
        x_grid, y_grid, z_pred_dog, alpha=0.3, cmap='Blues')

    # 设置标签
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP' if 'GDP' in title else 'Pet Economy')
    ax.set_zlabel('Number')

    # 设置标题
    ax.set_title(title)

    # 添加图例
    ax.legend()

    # 调整当前图形的布局
    plt.tight_layout()

# 显示所有图形
plt.show()
