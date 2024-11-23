import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
years = [2019, 2020, 2021, 2022, 2023]

# 判断量数据
cats_cn = [4412, 4862, 5806, 6536, 6980]
dogs_cn = [5503, 5222, 5429, 5119, 5175]
production = [441, 727, 1554, 1508, 2793]
exports = [154, 71, 89, 179, 287]

# 辅助量数据
us_cats = [9420, 6500, 9420, 7380, 7380]
us_dogs = [8970, 8500, 8970, 8970, 8010]
fr_cats = [1300, 1490, 1510, 1490, 1660]
fr_dogs = [740, 775, 750, 760, 990]
de_cats = [1470, 1570, 1670, 1520, 1570]
de_dogs = [1010, 1070, 1030, 1060, 1050]

# 计算判断量
pets_total = np.array(cats_cn) + np.array(dogs_cn)
judgment = np.array(production) + 5 * np.array(exports) + pets_total * 0.01

# 计算辅助量
auxiliary = np.array(us_cats) + np.array(us_dogs) + \
    np.array(fr_cats) + np.array(fr_dogs) + \
    np.array(de_cats) + np.array(de_dogs)

# 计算差分
diff_judgment = np.diff(judgment)
diff_auxiliary = np.diff(auxiliary)

# 计算影响量
impact = 0.5*diff_judgment - diff_auxiliary

# 绘制折线图
plt.figure(figsize=(5, 3))
plt.plot(years[1:], impact, marker='o', color='blue', linewidth=2)

# 添加零线
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 设置标题和标签
plt.title('Impact Change of Pet Food Industry')
plt.xlabel('Year')
plt.ylabel('Impact')

# 添加数据标签
for i, v in enumerate(impact):
    plt.text(years[i+1], v, f'{v:.2f}', ha='center', va='bottom')

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 优化布局
plt.tight_layout()

plt.show()
