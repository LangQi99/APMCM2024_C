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
years, data1 = read_data('q_1_back/data4.txt')
_, data2 = read_data('q_1_back/data5.txt')

# 准备数据
plots = [
    ('Predicted Production and Exports', years,
     data1['Production'], data1['Exports'], data2['demand']),
]

# 分别创建两个图形
for i, (title, x, cat, dog, y) in enumerate(plots):
    fig = plt.figure(figsize=(8, 8))  # 为每个图形创建新的fig
    ax = fig.add_subplot(111, projection='3d')  # 创建3D子图

    # 绘制散点图和回归平面
    ax.scatter(x, y, cat, color='red', label='Production')
    ax.scatter(x, y, dog, color='blue', label='Exports')

    # 准备回归数据（分别为猫和狗）
    X_cat = np.column_stack((x, y))
    X_dog = np.column_stack((x, y))

    regressor_cat = LinearRegression()
    regressor_dog = LinearRegression()

    regressor_cat.fit(X_cat, cat)  # 修改目标变量
    regressor_dog.fit(X_dog, dog)  # 修改目标变量

    # 生成网格数据 - 修改范围包含未来年份
    future_years = np.array([2024, 2025, 2026])

    # 准备未来年份的预测数据
    # future_y = np.interp(future_years, x, y)  # 对GDP/宠物经济进行插值
    future_y = np.array([3982670, 3735530, 3995310])
    future_X = np.column_stack((future_years, future_y))
    x_grid, y_grid = np.meshgrid(np.linspace(x.min(), max(future_years), 30),
                                 np.linspace(y.min(), max(future_y), 30))

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
    ax.set_ylabel('demand')
    ax.set_zlabel('Exports and Production')

    # 设置标题
    ax.set_title(title)

    # 添加图例
    ax.legend()

    # 添加2024-2026年的预测

    # 预测未来的猫和狗的数量
    future_cats = regressor_cat.predict(future_X)
    future_dogs = regressor_dog.predict(future_X)

    # 在3D图中添加预测点和标注
    for year, y_val, cat_pred, dog_pred in zip(future_years, future_y, future_cats, future_dogs):
        # 添加猫的预测点和标注
        ax.scatter(year, y_val, cat_pred, color='darkred', marker='*', s=100)
        ax.text(year, y_val, cat_pred,
                f'Year: {year}\nProduction: {cat_pred:.0f}',
                color='darkred', fontsize=8)

        # 添加狗的预测点和标注
        ax.scatter(year, y_val, dog_pred, color='darkblue', marker='*', s=100)
        ax.text(year, y_val, dog_pred,
                f'Year: {year}\nExports: {dog_pred:.0f}',
                color='darkblue', fontsize=8)

    # 添加图例
    ax.scatter([], [], color='darkred', marker='*',
               s=100, label='Predicted Production')
    ax.scatter([], [], color='darkblue', marker='*',
               s=100, label='Predicted Exports')
    ax.legend()

    # 打印预测结果
    print(f"\n预测结果 - {title}:")
    for year, y_val, cat_pred, dog_pred in zip(future_years, future_y, future_cats, future_dogs):
        print(f"年份: {year}")
        print(f"{'GDP' if 'GDP' in title else '宠物经济'}: {y_val:.2f}")
        print(f"预测猫的数量: {cat_pred:.2f}")
        print(f"预测狗的数量: {dog_pred:.2f}")
        print("-" * 30)

    # 在所有子图绘制完成后调整整体布局
    plt.tight_layout()

    # 显示当前图形
    plt.show()  # 在每个图形后显示
