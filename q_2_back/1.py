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

# 创建一个包含三个子图的图表
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
fig.suptitle('Cat and Dog Numbers', fontsize=16)

# 为每个国家创建子图
for i, (country, file_path) in enumerate(countries.items()):
    years, cat_data, dog_data = read_data_file(file_path)

    axes[i].plot(years, cat_data, marker='o', label='Cat')
    axes[i].plot(years, dog_data, marker='o', label='Dog')

    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Number')
    axes[i].set_title(f'{country}')
    axes[i].legend()
    axes[i].grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 保存整个图表
plt.savefig('q_2/ori/all_countries_pets.png')
plt.close()
