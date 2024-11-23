import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy import stats  # 添加这个import用于QQ图
from statsmodels.stats.diagnostic import acorr_ljungbox  # 添加这个import
from statsmodels.graphics.tsaplots import plot_acf  # 添加这个import

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

    # 调整years以匹配残差长度
    adjusted_years = years[2:]  # 因为ARIMA(1,1,1)会损失2个数据点

    # 创建一个2x2的子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 绘制猫的残差图
    ax1.plot(adjusted_years, cat_resid, 'o-', label='Residuals')
    ax1.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    ax1.set_title(f'{country} - Cat ARIMA Residuals')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Residual Value')
    ax1.legend()

    # 绘制狗的残差图
    ax2.plot(adjusted_years, dog_resid, 'o-', label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    ax2.set_title(f'{country} - Dog ARIMA Residuals')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Residual Value')
    ax2.legend()

    # 对ACF图使用maxlags参数来限制滞后阶数
    max_lags = min(10, len(cat_resid) - 1)  # 确保滞后阶数不超过数据长度
    plot_acf(cat_resid, lags=max_lags, ax=ax3,
             title=f'{country} - Cat Residuals ACF')
    plot_acf(dog_resid, lags=max_lags, ax=ax4,
             title=f'{country} - Dog Residuals ACF')

    # 进行Ljung-Box检验
    cat_lb_stat, cat_lb_p = acorr_ljungbox(
        cat_resid, lags=[10], return_df=False)
    dog_lb_stat, dog_lb_p = acorr_ljungbox(
        dog_resid, lags=[10], return_df=False)

    # 在图中添加Ljung-Box检验结果
    ax3.text(0.05, 0.95,
             f'Ljung-Box test p-value: {cat_lb_p[0]:.4f}',
             transform=ax3.transAxes, verticalalignment='top')
    ax4.text(0.05, 0.95,
             f'Ljung-Box test p-value: {dog_lb_p[0]:.4f}',
             transform=ax4.transAxes, verticalalignment='top')

    # 进行Shapiro-Wilk正态性检验和Jarque-Bera检验
    cat_shapiro_stat, cat_shapiro_p = stats.shapiro(cat_resid)
    dog_shapiro_stat, dog_shapiro_p = stats.shapiro(dog_resid)

    cat_jb_stat, cat_jb_p = stats.jarque_bera(cat_resid)
    dog_jb_stat, dog_jb_p = stats.jarque_bera(dog_resid)

    # 添加检验结果文本
    ax1.text(0.05, 0.95,
             f'Shapiro-Wilk test p-value: {cat_shapiro_p:.4f}\n'
             f'Jarque-Bera test p-value: {cat_jb_p:.4f}',
             transform=ax1.transAxes, verticalalignment='top')
    ax2.text(0.05, 0.95,
             f'Shapiro-Wilk test p-value: {dog_shapiro_p:.4f}\n'
             f'Jarque-Bera test p-value: {dog_jb_p:.4f}',
             transform=ax2.transAxes, verticalalignment='top')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(f'q_2/pri1/{country.lower()}_residuals2.png')
    plt.close()
