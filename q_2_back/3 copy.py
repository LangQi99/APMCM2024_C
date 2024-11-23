import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 创建pri1文件夹（如果不存在）
if not os.path.exists('q_2/pri1'):
    os.makedirs('q_2/pri1')

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

    # 创建ARIMA模型并预测
    future_years = list(range(years[0] + 1, years[0] + 4))  # 预测未来3年
    all_years = years + future_years

    # 对猫的数据进行预测
    cat_model = ARIMA(cat_data, order=(1, 1, 1))
    cat_fit = cat_model.fit()
    cat_forecast = cat_fit.forecast(steps=3)
    cat_all_data = np.concatenate([cat_data, cat_forecast])

    # 对狗的数据进行预测
    dog_model = ARIMA(dog_data, order=(1, 1, 1))
    dog_fit = dog_model.fit()
    dog_forecast = dog_fit.forecast(steps=3)
    dog_all_data = np.concatenate([dog_data, dog_forecast])

    # 绘制图表
    plt.figure(figsize=(12, 7))

    # 绘制历史数据（实线）
    plt.plot(years, cat_data, marker='o',
             label='Cat', solid_capstyle='round')
    plt.plot(years, dog_data, marker='o',
             label='Dog', solid_capstyle='round')

    # 绘制预测数据（虚线）并连接历史数据和预测数据
    plt.plot(future_years, cat_forecast, 'o-', label='Predicted Cat')
    plt.plot([years[0], future_years[0]],
             [cat_data[0], cat_forecast[0]], 'b--')  # 连接猫的历史和预测数据

    plt.plot(future_years, dog_forecast, 'o-', label='Predicted Dog')
    plt.plot([years[0], future_years[0]], [dog_data[0],
             dog_forecast[0]], 'C1--')  # 连接狗的历史和预测数据

    # 添加数值标签
    # for i, txt in enumerate(cat_data):
    #     plt.annotate(str(txt), (years[i], cat_data[i]), textcoords="offset points", xytext=(
    #         0, 10), ha='center')
    # for i, txt in enumerate(dog_data):
    #     plt.annotate(str(txt), (years[i], dog_data[i]), textcoords="offset points", xytext=(
    #         0, -15), ha='center')

    # 添加预测值标签
    for i, txt in enumerate(cat_forecast):
        plt.annotate(f'{txt:.0f}', (future_years[i], txt), textcoords="offset points", xytext=(
            0, 10), ha='center')
    for i, txt in enumerate(dog_forecast):
        plt.annotate(f'{txt:.0f}', (future_years[i], txt), textcoords="offset points", xytext=(
            0, -15), ha='center')

    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.title(f'{country} - Cat and Dog Numbers with Forecast')
    plt.legend()
    plt.grid(True)

    # 保存图表到pri1文件夹
    plt.savefig(f'q_2/pri1/{country.lower()}_pets_forecast.png')
    plt.close()
