import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('data_2.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
years = lines[0].split()[1:]  # 获取年份
data = lines[1].split()[1:]  # 获取数据

# 将字符串转换为数值
years = [int(year) for year in years]
data = [int(num) for num in data]

# 数据预处理：对数变换
min_year = min(years)
years_log = [year - min_year + 1 for year in years]
data_log = [d - 10000 for d in data]
years_log = np.log(years_log)

# 多项式拟合循环
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 对数变换后的拟合
    coeffs = np.polyfit(years_log, data_log, n)
    y_poly_pred = np.polyval(coeffs, years_log)

    # 预测新值（对数空间）
    new_x = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    new_x_log = np.log(new_x - min_year + 1)
    new_y_log = np.polyval(coeffs, new_x_log)

    # 第一个子图：对数变换后的数据
    ax1.plot(years_log, data_log, marker='o', label='Transformed Pet Economy')
    ax1.plot(new_x_log, new_y_log, marker='o',
             linestyle='--', label='Predicted')
    ax1.set_title(f'Log-transformed Data ({n} order polynomial)')
    ax1.legend()
    ax1.grid(True)

    # 转换回原始空间
    y_poly_pred_original = [y + 10000 for y in y_poly_pred]
    new_y_original = [y + 10000 for y in new_y_log]

    # 第二个子图：原始空间的数据
    ax2.plot(years, data, marker='o', label='Pet Economy')
    ax2.plot(new_x, new_y_original, marker='o',
             linestyle='--', label='Predicted')

    # 添加预测值标注
    for i, (x, y) in enumerate(zip(new_x, new_y_original)):
        ax2.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    ax2.set_title(f'Original Data Space ({n} order polynomial)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'question1/qlog2_{n}.png')
    plt.close()
