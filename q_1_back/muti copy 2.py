import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# 准备合并后的数据
X = np.column_stack(
    (years, data2['GDP'], data3['pet_industry_economy']))  # 年份、GDP、宠物经济
y_cat = data1['cat']
y_dog = data1['dog']

# 创建多变量线性回归模型
regressor_cat = LinearRegression()
regressor_dog = LinearRegression()

# 训练模型
regressor_cat.fit(X, y_cat)
regressor_dog.fit(X, y_dog)

# 准备未来预测数据
future_years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
future_gdp = np.array([10142, 10407, 12617, 12662, 12613, 13479, 15640, 19190])
future_pet_economy = np.array([2112, 2407, 2703, 2998, 3294, 3589, 3885, 4181])
future_X = np.column_stack((future_years, future_gdp, future_pet_economy))

# 预测未来值
future_cats = regressor_cat.predict(future_X)
future_dogs = regressor_dog.predict(future_X)

# 创建2D可视化（更新后的部分）
plt.figure(figsize=(10, 6))
plt.plot(years, y_cat, marker='o', label='Cat')
plt.plot(years, y_dog, marker='o', label='Dog')
plt.plot(future_years, future_cats, marker='o',
         linestyle='--', label='Predicted Cat')
plt.plot(future_years, future_dogs, marker='o',
         linestyle='--', label='Predicted Dog')

# 添加预测值标签（更新后的标注样式）
for year, cat in zip(future_years, future_cats):
    plt.annotate(f'{int(cat)}', (year, cat), textcoords="offset points",
                 xytext=(0, 10), ha='center')

for year, dog in zip(future_years, future_dogs):
    plt.annotate(f'{int(dog)}', (year, dog), textcoords="offset points",
                 xytext=(0, -15), ha='center')

# 设置图表属性
plt.xlabel('Year')
plt.ylabel('Number')
plt.title(
    'Predicted Cat and Dog Numbers Over Years\n(Based on Year, GDP, and Pet Economy)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 打印预测结果和模型系数
print("\n预测结果:")
for year, gdp, pet_eco, cat_pred, dog_pred in zip(future_years, future_gdp, future_pet_economy, future_cats, future_dogs):
    print(f"年份: {year}")
    print(f"GDP: {gdp:.2f}")
    print(f"宠物经济: {pet_eco:.2f}")
    print(f"预测猫的数量: {cat_pred:.2f}")
    print(f"预测狗的数量: {dog_pred:.2f}")
    print("-" * 30)

# 打印模型系数
print("\n模型系数:")
print("猫的模型:")
print(f"Year coefficient: {regressor_cat.coef_[0]:.4f}")
print(f"GDP coefficient: {regressor_cat.coef_[1]:.4f}")
print(f"Pet Economy coefficient: {regressor_cat.coef_[2]:.4f}")
print(f"Intercept: {regressor_cat.intercept_:.4f}")

print("\n狗的模型:")
print(f"Year coefficient: {regressor_dog.coef_[0]:.4f}")
print(f"GDP coefficient: {regressor_dog.coef_[1]:.4f}")
print(f"Pet Economy coefficient: {regressor_dog.coef_[2]:.4f}")
print(f"Intercept: {regressor_dog.intercept_:.4f}")

plt.show()
