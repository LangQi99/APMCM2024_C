import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 创建pri2文件夹（如果不存在）
if not os.path.exists('q_2/pri2'):
    os.makedirs('q_2/pri2')

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


countries = {
    'America': 'q_2/america.txt',
    # 'France': 'q_2/france.txt',
    # 'Germany': 'q_2/germany.txt'
}

# 为每个国家创建单独的图表
for country, file_path in countries.items():
    years, cat_data, dog_data = read_data_file(file_path)

    # 创建一个2x3的图表布局
    fig = plt.figure(figsize=(18, 10))

    # 主预测图
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    # 创建ARIMA模型并预测
    future_years = list(range(years[-1] + 1, years[-1] + 4))  # 使用最后一年作为基准
    all_years = years + future_years

    # 对猫的数据进行预测
    cat_data_array = np.array(cat_data)  # 转换为numpy数组
    dog_data_array = np.array(dog_data)  # 转换为numpy数组

    # 对数据进行差分处理以获得更好的平稳性
    cat_diff = np.diff(cat_data_array)
    dog_diff = np.diff(dog_data_array)

    # 修改ARIMA模型参数
    cat_model = ARIMA(cat_data_array, order=(1, 1, 1))  # 减少差分阶数
    cat_fit = cat_model.fit()
    # 修改预测起始点为第一年
    cat_pred = cat_fit.get_prediction(start=0, end=len(all_years)-1)
    cat_predicted = cat_pred.predicted_mean
    cat_forecast = cat_predicted[len(years):]
    cat_all_data = cat_predicted

    # 对狗的数据进行预测
    dog_model = ARIMA(dog_data_array, order=(1, 1, 1))  # 减少差分阶数
    dog_fit = dog_model.fit()
    # 同样修改狗的预测起始点为第一年
    dog_pred = dog_fit.get_prediction(start=0, end=len(all_years)-1)
    dog_predicted = dog_pred.predicted_mean
    dog_forecast = dog_predicted[len(years):]
    dog_all_data = dog_predicted

    # 在主图(ax1)上绘制预测
    ax1.plot(all_years[:len(years)], cat_data, 'o-',
             label='Cat', solid_capstyle='round')
    ax1.plot(all_years[:len(years)], dog_data, 'o-',
             label='Dog', solid_capstyle='round')
    ax1.plot(all_years, cat_all_data, 'o--', label='Predicted Cat')
    ax1.plot(all_years, dog_all_data, 'o--', label='Predicted Dog')

    # 添加标签
    for i, txt in enumerate(cat_all_data):
        ax1.annotate(f'{txt:.0f}', (all_years[i], txt),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    for i, txt in enumerate(dog_all_data):
        ax1.annotate(f'{txt:.0f}', (all_years[i], txt),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number')
    ax1.set_title(f'{country} - Predicted Cat and Dog Numbers')
    ax1.legend()
    ax1.grid(True)

    # 添加平稳性分析
    # Cat数据的ACF
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    plot_acf(cat_data_array, ax=ax2, title=f'Cat ACF')

    # Cat数据的PACF
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    plot_pacf(cat_data_array, ax=ax3, title=f'Cat PACF')

    # Dog数据的ACF
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    plot_acf(dog_data_array, ax=ax4, title=f'Dog ACF')

    # Dog数据的PACF
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    plot_pacf(dog_data_array, ax=ax5, title=f'Dog PACF')

    # 进行ADF测试并添加结果文本
    cat_adf = adfuller(cat_data)
    dog_adf = adfuller(dog_data)

    plt.figtext(0.02, 0.02, f'Cat ADF p-value: {cat_adf[1]:.4f}\nDog ADF p-value: {dog_adf[1]:.4f}',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(
        f'q_2/pri2/{country.lower()}_pets_forecast.png', bbox_inches='tight')
    plt.close()
