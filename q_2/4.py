import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy import stats  # 添加这个import用于QQ图

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
    cat_model = ARIMA(cat_data, order=(1, 1, 1))
    cat_fit = cat_model.fit()
    cat_resid = cat_fit.resid  # 获取残差

    dog_model = ARIMA(dog_data, order=(1, 1, 1))
    dog_fit = dog_model.fit()
    dog_resid = dog_fit.resid  # 获取残差

    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制猫的QQ图
    stats.probplot(cat_resid, dist="norm", plot=ax1)
    ax1.set_title(f'{country} - Cat ARIMA Residuals Q-Q Plot')

    # 绘制狗的QQ图
    stats.probplot(dog_resid, dist="norm", plot=ax2)
    ax2.set_title(f'{country} - Dog ARIMA Residuals Q-Q Plot')

    # 调整布局
    plt.tight_layout()

    # 保存图表到pri1文件夹
    plt.savefig(f'q_2/pri1/{country.lower()}_qq_plots.png')
    plt.close()
