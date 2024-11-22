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

    # 反转数据顺序，使年份从小到大排序
    years.reverse()
    cat_data.reverse()
    dog_data.reverse()
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
    future_years = list(range(years[-1] + 1, years[-1] + 4))  # 使用最后一年作为基准
    all_years = years + future_years

    # 对猫的数据进行预测
    cat_model = ARIMA(cat_data, order=(1, 1, 1))
    cat_fit = cat_model.fit()
    # 修改预测起始点为第一个观测值
    cat_pred = cat_fit.get_prediction(start=1, end=len(all_years)-1)
    cat_predicted = np.concatenate(
        ([cat_data[0]], cat_pred.predicted_mean[:len(years)-1]))
    cat_forecast = cat_pred.predicted_mean[len(years)-1:]
    cat_all_data = np.concatenate((cat_predicted, cat_forecast))

    # 对狗的数据进行预测
    dog_model = ARIMA(dog_data, order=(1, 1, 1))
    dog_fit = dog_model.fit()
    # 同样修改狗的预测起始点
    dog_pred = dog_fit.get_prediction(start=1, end=len(all_years)-1)
    dog_predicted = np.concatenate(
        ([dog_data[0]], dog_pred.predicted_mean[:len(years)-1]))
    dog_forecast = dog_pred.predicted_mean[len(years)-1:]
    dog_all_data = np.concatenate((dog_predicted, dog_forecast))

    # 绘制图表
    plt.figure(figsize=(12, 7))

    # 绘制实际数据（实线）
    plt.plot(all_years[:len(years)], cat_data, 'o-',
             label='Cat Actual', solid_capstyle='round')
    plt.plot(all_years[:len(years)], dog_data, 'o-',
             label='Dog Actual', solid_capstyle='round')

    # 绘制拟合和预测数据（虚线）
    plt.plot(all_years, cat_all_data, 'o--',
             label='Cat Fitted & Predicted')
    plt.plot(all_years, dog_all_data, 'o--',
             label='Dog Fitted & Predicted')

    # 添加所有预测值标签（包括拟合值和预测值）
    for i, txt in enumerate(cat_all_data):
        plt.annotate(f'{txt:.0f}', (all_years[i], txt),
                     textcoords="offset points",
                     xytext=(0, 10), ha='center')
    for i, txt in enumerate(dog_all_data):
        plt.annotate(f'{txt:.0f}', (all_years[i], txt),
                     textcoords="offset points",
                     xytext=(0, 10), ha='center')  # 向下偏移以避免重叠

    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.title(f'{country} - Predicted Cat and Dog Numbers')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig(f'q_2/pri1/{country.lower()}_pets_forecast.png')
    plt.close()
