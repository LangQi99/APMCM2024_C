import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('q_1/data_2.txt', 'r') as file:
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

# 创建一个4x2的大图布局（而不是2x4）
fig, axes = plt.subplots(2, 4, figsize=(15, 6))  # 调整了figsize以适应新的布局

# 多项式拟合循环
for n in [1, 2, 3, 4]:
    # 获取当前子图位置
    ax1 = axes[0, n-1]  # 第一列
    ax2 = axes[1, n-1]  # 第二列

    # 对数变换后的拟合
    coeffs = np.polyfit(years_log, data_log, n)
    y_poly_pred = np.polyval(coeffs, years_log)

    # 预测新值（对数空间）
    new_x = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    new_x_log = np.log(new_x - min_year + 1)
    new_y_log = np.polyval(coeffs, new_x_log)

    # 第一个子图：对数变换后的数据
    ax1.plot(years_log, data_log, marker='o', label='Transformed GDP')
    ax1.plot(new_x_log, new_y_log, marker='o',
             linestyle='--', label='Predicted')
    ax1.set_title(f'{n} order polynomial (Log-transformed)')
    ax1.legend()
    ax1.grid(True)

    # 转换回原始空间
    y_poly_pred_original = [y + 10000 for y in y_poly_pred]
    new_y_original = [y + 10000 for y in new_y_log]

    # 第二个子图：原始空间的数据
    ax2.plot(years, data, marker='o', label='GDP')
    ax2.plot(new_x, new_y_original, marker='o',
             linestyle='--', label='Predicted')

    # 添加预测值标注
    for i, (x, y) in enumerate(zip(new_x, new_y_original)):
        ax2.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    ax2.set_title(f'{n} order polynomial (Original)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number')
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
plt.savefig('q_1/question1/qlog_all.png')
plt.close()
