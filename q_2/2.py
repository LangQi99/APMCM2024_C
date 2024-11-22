import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建ori文件夹（如果不存在）
if not os.path.exists('q_2/ori'):
    os.makedirs('q_2/ori')

# 定义函数读取数据文件


def read_data_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    years = [int(year) for year in lines[0].split()[1:]]
    cat_data = [int(num) for num in lines[1].split()[1:]]
    dog_data = [int(num) for num in lines[2].split()[1:]]
    return years, cat_data, dog_data


# 读取三个国家的数据
countries = {
    'America': 'q_2/america.txt',
    'France': 'q_2/france.txt',
    'Germany': 'q_2/germany.txt'
}

# 创建一个大图表
plt.figure(figsize=(12, 7))

# 为不同国家使用不同的线型和颜色
line_styles = {
    'America': ('-', 'red'),
    'France': ('-', 'blue'),
    'Germany': ('-', 'green')
}

# 读取并绘制所有国家的数据
for country, file_path in countries.items():
    years, cat_data, dog_data = read_data_file(file_path)

    style, color = line_styles[country]
    # 绘制猫的数据
    plt.plot(years, cat_data, style, color=color, marker='o',
             label=f'{country} - Cat')
    # 绘制狗的数据
    plt.plot(years, dog_data, style, color=color, marker='s',
             label=f'{country} - Dog')

plt.xlabel('Year')
plt.ylabel('Number')
plt.title('Comparison of Cat and Dog Numbers Across Countries')
plt.legend()
plt.grid(True)

# 保存合并后的图表
plt.savefig('q_2/ori/combined_pets.png')
plt.close()
