import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('q_1_back/data_2.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
years = lines[0].split()[1:]  # 获取年份
data = lines[1].split()[1:]  # 获取数据

# 将字符串转换为数值
years = [int(year) for year in years]
data = [int(num) for num in data]

# 多项式拟合（例如：2次多项式）
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    coeffs = np.polyfit(years, data, n)

    # 使用多项式预测原始 x 值对应的 y 值
    y_poly_pred = np.polyval(coeffs, years)

    # 新的 x 值用于预测
    new_x = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])

    # 使用拟合的多项式预测新的 y 值
    new_y = np.polyval(coeffs, new_x)

    # 创建折线图
    plt.figure(figsize=(6, 4))
    plt.plot(years, data, marker='o', label='GDP')
    # 绘制原始数据和拟合曲线
    # plt.scatter(years, cat_data, color='blue', label='Data points')
    # plt.plot(years, y_poly_pred, color='green', label='Polynomial fit')

    # 绘制新的预测点 - 使用虚线
    plt.plot(new_x, new_y, marker='o',
             linestyle='--', label='Predicted')

    # 添加预测值标注
    for i, (x, y) in enumerate(zip(new_x, new_y)):
        plt.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    # 设置图表属性
    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.title(
        f'Predicted GDP Over Years({n} order polynomial)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q_1_back/question1_/q2_{n}_.png')
    # 设置y轴范围，比数据的最大值稍大一些
    # y_max = max(max(cat_data), max(dog_data))
    # plt.ylim(0, y_max * 1.2)  # 将最大值扩大20%

    # 反转x轴（因为原数据年份是从大到小）
    # plt.gca().invert_xaxis()

    # 显示图表

    # plt.show()
