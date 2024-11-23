import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
years = [2019, 2020, 2021, 2022, 2023]
cats = [4412, 4862, 5806, 6536, 6980]
dogs = [5503, 5222, 5429, 5119, 5175]
production = [441, 727, 1554, 1508, 2793]
exports = [154, 71, 89, 179, 287]

# 计算判断量
pets_total = np.array(cats) + np.array(dogs)
production_arr = np.array(production)
exports_arr = np.array(exports)
judgment = production_arr + 5*exports_arr + pets_total * 0.01

# 计算差分
diff_judgment = np.diff(judgment)
diff_years = years[1:]

# 创建图形
fig, ax1 = plt.subplots(figsize=(5, 3))

# 绘制柱状图
bars = ax1.bar(years, judgment, color='skyblue',
               alpha=0.7, label='Judgment Value')
ax1.set_xlabel('Year')
ax1.set_ylabel('Judgment Value', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Create secondary y-axis
ax2 = ax1.twinx()

# Plot difference line graph
line = ax2.plot(diff_years, diff_judgment,
                color='red', marker='o', label='Difference Value')
ax2.set_ylabel('Difference Value', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Judgment Value and Its Difference Change')
plt.tight_layout()
plt.show()
