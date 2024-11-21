import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('data_1.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
years = lines[0].split()[1:]  # 获取年份
cat_data = lines[1].split()[1:]  # 获取猫的数据
dog_data = lines[2].split()[1:]  # 获取狗的数据

# 将字符串转换为数值
years = [int(year) for year in years]
cat_data = [int(num) for num in cat_data]
dog_data = [int(num) for num in dog_data]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(years, cat_data, marker='o', label='Cat')
plt.plot(years, dog_data, marker='o', label='Dog')

# 设置图表属性
plt.xlabel('Year')
plt.ylabel('Number')
plt.title('Cat and Dog Numbers Over Years')
plt.legend()
plt.grid(True)

# 设置y轴范围，比数据的最大值稍大一些
# y_max = max(max(cat_data), max(dog_data))
# plt.ylim(0, y_max * 1.2)  # 将最大值扩大20%

# 反转x轴（因为原数据年份是从大到小）
# plt.gca().invert_xaxis()

# 显示图表
plt.show()
