import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 创建pri2文件夹（如果不存在）
if not os.path.exists('q_2_amer/pri2'):
    os.makedirs('q_2_amer/pri2')

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
    'America': 'q_2_amer/america.txt',
    # 'France': 'q_2/france.txt',
    # 'Germany': 'q_2/germany.txt'
}

# 为每个国家创建单独的图表
for country, file_path in countries.items():
    years, cat_data, dog_data = read_data_file(file_path)

    # 枚举不同的差分次数
    for k in range(3):  # k = 0, 1, 2
        # 创建一个2x3的图表布局
        fig = plt.figure(figsize=(12, 7))

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
        cat_model = ARIMA(cat_data_array, order=(1, k, 1))  # 减少差分阶数
        cat_fit = cat_model.fit()
        # 修改预测起始点为第一个观测值
        cat_pred = cat_fit.get_prediction(start=1, end=len(all_years)-1)
        cat_predicted = np.concatenate(
            ([cat_data[0]], cat_pred.predicted_mean[:len(years)-1]))
        cat_forecast = cat_pred.predicted_mean[len(years)-1:]
        cat_all_data = np.concatenate((cat_predicted, cat_forecast))

        # 对狗的数据进行预测
        dog_model = ARIMA(dog_data_array, order=(1, k, 1))  # 减少差分阶数
        dog_fit = dog_model.fit()
        # 同样修改狗的预测起始点
        dog_pred = dog_fit.get_prediction(start=1, end=len(all_years)-1)
        dog_predicted = np.concatenate(
            ([dog_data[0]], dog_pred.predicted_mean[:len(years)-1]))
        dog_forecast = dog_pred.predicted_mean[len(years)-1:]
        dog_all_data = np.concatenate((dog_predicted, dog_forecast))

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

        # 替换残差图部分
        # 猫的残差柱状图
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        # 排除第一年的残差
        ax2.bar(years[1:], cat_fit.resid[1:])
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Cat Model Residuals (Excluding First Year)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Residual')

        # 猫的残差Q-Q图 (排除第一年)
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        from scipy import stats
        stats.probplot(cat_fit.resid[1:], dist="norm", plot=ax3)
        ax3.set_title('Cat Residuals Q-Q Plot (Excluding First Year)')

        # 狗的残差柱状图
        ax4 = plt.subplot2grid((2, 3), (1, 1))
        ax4.bar(years[1:], dog_fit.resid[1:])
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('Dog Model Residuals (Excluding First Year)')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Residual')

        # 狗的残差Q-Q图 (排除第一年)
        ax5 = plt.subplot2grid((2, 3), (1, 2))
        stats.probplot(dog_fit.resid[1:], dist="norm", plot=ax5)
        ax5.set_title('Dog Residuals Q-Q Plot (Excluding First Year)')

        # 计算模型评估指标（排除第一年）
        cat_mse = np.mean(cat_fit.resid[1:]**2)
        dog_mse = np.mean(dog_fit.resid[1:]**2)
        total_mse = cat_mse + dog_mse

        # 打印评估结果
        print(f'\n差分阶数 k={k} 的评估结果：')
        print(f'猫模型MSE: {cat_mse:.2f}')
        print(f'狗模型MSE: {dog_mse:.2f}')
        print(f'总体MSE: {total_mse:.2f}')

        plt.tight_layout()
        plt.savefig(
            f'q_2_amer/pri2/{country.lower()}_pets_forecast_k{k}.png', bbox_inches='tight')
        plt.close()
