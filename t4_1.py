import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
years = [2019, 2020, 2021, 2022, 2023]

# 美国数据
us_cats = [9420, 6500, 9420, 7380, 7380]
us_dogs = [8970, 8500, 8970, 8970, 8010]

# 法国数据
fr_cats = [1300, 1490, 1510, 1490, 1660]
fr_dogs = [740, 775, 750, 760, 990]

# 德国数据
de_cats = [1470, 1570, 1670, 1520, 1570]
de_dogs = [1010, 1070, 1030, 1060, 1050]

# 计算辅助量（各国猫狗总和）
auxiliary = np.array(us_cats) + np.array(us_dogs) + \
    np.array(fr_cats) + np.array(fr_dogs) + \
    np.array(de_cats) + np.array(de_dogs)

# 计算差分
diff_auxiliary = np.diff(auxiliary)
diff_years = years[1:]

# 创建图形
fig, ax1 = plt.subplots(figsize=(5, 3))

# 绘制柱状图
bars = ax1.bar(years, auxiliary, color='lightgreen',
               alpha=0.7, label='Auxiliary Value')
ax1.set_xlabel('Year')
ax1.set_ylabel('Auxiliary Value', color='lightgreen')
ax1.tick_params(axis='y', labelcolor='lightgreen')

# Create secondary y-axis
ax2 = ax1.twinx()

# Plot difference line graph
line = ax2.plot(diff_years, diff_auxiliary,
                color='purple', marker='o', label='Difference Value')
ax2.set_ylabel('Difference Value', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Auxiliary Value and Its Difference Change')
plt.tight_layout()
plt.show()
