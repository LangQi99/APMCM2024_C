import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('q_1/data_1.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
years = lines[0].split()[1:]  # 获取年份
cat_data = lines[1].split()[1:]  # 获取猫的数据
dog_data = lines[2].split()[1:]  # 获取狗的数据

# 将字符串转换为数值
years = [int(year) for year in years]
cat_data = [int(num) for num in cat_data]
dog_data = [int(num) for num in dog_data]

# 创建一个4x2的子图网格（而不是3x3）
plt.figure(figsize=(10, 15))  # 调整图形大小以适应新的布局

for n in range(2, 10):
    # 在4x2网格中创建子图
    plt.subplot(4, 2, n-1)  # 修改子图位置计算

    coeffs_cat = np.polyfit(years, cat_data, n)
    coeffs_dog = np.polyfit(years, dog_data, n)

    # 使用多项式预测原始 x 值对应的 y 值
    y_poly_pred_cat = np.polyval(coeffs_cat, years)
    y_poly_pred_dog = np.polyval(coeffs_dog, years)

    # 新的 x 值用于预测
    new_x = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])

    # 使用拟合的多项式预测新的 y 值
    new_y_cat = np.polyval(coeffs_cat, new_x)
    new_y_dog = np.polyval(coeffs_dog, new_x)

    # 绘制到当前子图
    plt.plot(years, cat_data, marker='o', label='Cat')
    plt.plot(years, dog_data, marker='o', label='Dog')
    plt.plot(new_x, new_y_cat, marker='o',
             linestyle='--', label='Predicted Cat')
    plt.plot(new_x, new_y_dog, marker='o',
             linestyle='--', label='Predicted Dog')

    # 添加预测值标注（字体大小调小）
    for i, (x, y) in enumerate(zip(new_x, new_y_cat)):
        plt.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)

    for i, (x, y) in enumerate(zip(new_x, new_y_dog)):
        plt.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=8)

    # 设置每个子图的属性
    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.title(f'Order {n} polynomial')
    plt.legend(fontsize=8)
    plt.grid(True)

# 调整子图之间的间距
plt.tight_layout()
# 保存整个图表
plt.savefig('q_1/question1/q1_all.png')
# plt.show()
