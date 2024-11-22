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

# 为每个国家创建单独的图表
for country, file_path in countries.items():
    years, cat_data, dog_data = read_data_file(file_path)

    plt.figure(figsize=(10, 6))
    plt.plot(years, cat_data, marker='o', label='Cat')
    plt.plot(years, dog_data, marker='o', label='Dog')

    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.title(f'{country} - Cat and Dog Numbers Over Years')
    plt.legend()
    plt.grid(True)

    # 保存图表到ori文件夹
    plt.savefig(f'q_2/ori/{country.lower()}_pets.png')
    plt.close()
