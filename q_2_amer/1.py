import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 创建ori文件夹（如果不存在）
if not os.path.exists('q_2_amer/ori'):
    os.makedirs('q_2_amer/ori')

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
    'America': 'q_2_amer/america.txt',
    'France': 'q_2/france.txt',
    'Germany': 'q_2/germany.txt'
}

# 计算i阶差分


def calculate_diff(data, order=1):
    diff_data = data.copy()
    for _ in range(order):
        diff_data = [diff_data[i+1] - diff_data[i]
                     for i in range(len(diff_data)-1)]
    return diff_data


# 在文件开始处添加输出文件的设置
output_file = open('q_2_amer/analysis_results.txt', 'w', encoding='utf-8')

# 为每个国家创建差分图表
for country, file_path in countries.items():
    years, cat_data, dog_data = read_data_file(file_path)

    # 计算0-2阶差分
    for i in range(0, 3):
        plt.figure(figsize=(10, 6))

        # 计算猫和狗数据的i阶差分
        cat_diff = calculate_diff(cat_data, i)
        dog_diff = calculate_diff(dog_data, i)
        # 对应的年份需要减少i个数据点
        diff_years = years[i:]

        plt.plot(diff_years, cat_diff, marker='o', label='Cat')
        plt.plot(diff_years, dog_diff, marker='o', label='Dog')

        plt.xlabel('Year')
        plt.ylabel(f'{i}-order Difference')
        plt.title(f'{country} - {i}-order Difference of Cat and Dog Numbers')
        plt.legend()
        plt.grid(True)

        # 保存差分图表
        plt.savefig(f'q_2_amer/ori/{country.lower()}_pets_diff{i}.png')
        plt.close()

        # 创建ACF和PACF图
        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{country} - {i}-order Difference ACF and PACF')

        # 计算合适的nlags值（不超过样本大小的50%）
        max_lags = min(10, len(cat_diff) // 2)  # 取10和样本大小一半中的较小值

        # 计算猫数据的ACF和PACF
        cat_acf = acf(cat_diff, nlags=max_lags)
        cat_pacf = pacf(cat_diff, nlags=max_lags)
        lags = range(len(cat_acf))

        # 绘制猫数据的ACF
        ax1[0].stem(lags, cat_acf)
        ax1[0].set_title(f'Cat ACF (diff={i})')
        ax1[0].axhline(y=0, linestyle='--', color='gray')
        ax1[0].axhline(y=1.96/np.sqrt(len(cat_diff)),
                       linestyle='--', color='gray')
        ax1[0].axhline(y=-1.96/np.sqrt(len(cat_diff)),
                       linestyle='--', color='gray')

        # 绘制猫数据的PACF
        ax1[1].stem(lags, cat_pacf)
        ax1[1].set_title(f'Cat PACF (diff={i})')
        ax1[1].axhline(y=0, linestyle='--', color='gray')
        ax1[1].axhline(y=1.96/np.sqrt(len(cat_diff)),
                       linestyle='--', color='gray')
        ax1[1].axhline(y=-1.96/np.sqrt(len(cat_diff)),
                       linestyle='--', color='gray')

        # 计算狗数据的ACF和PACF
        dog_acf = acf(dog_diff, nlags=max_lags)
        dog_pacf = pacf(dog_diff, nlags=max_lags)

        # 绘制狗数据的ACF
        ax2[0].stem(lags, dog_acf)
        ax2[0].set_title(f'Dog ACF (diff={i})')
        ax2[0].axhline(y=0, linestyle='--', color='gray')
        ax2[0].axhline(y=1.96/np.sqrt(len(dog_diff)),
                       linestyle='--', color='gray')
        ax2[0].axhline(y=-1.96/np.sqrt(len(dog_diff)),
                       linestyle='--', color='gray')

        # 绘制狗数据的PACF
        ax2[1].stem(lags, dog_pacf)
        ax2[1].set_title(f'Dog PACF (diff={i})')
        ax2[1].axhline(y=0, linestyle='--', color='gray')
        ax2[1].axhline(y=1.96/np.sqrt(len(dog_diff)),
                       linestyle='--', color='gray')
        ax2[1].axhline(y=-1.96/np.sqrt(len(dog_diff)),
                       linestyle='--', color='gray')

        plt.tight_layout()
        plt.savefig(f'q_2_amer/ori/{country.lower()}_acf_pacf_diff{i}.png')
        plt.close()

        # 进行ADF检验
        print(f"\n{country} {i}阶差分的ADF检验结果：", file=output_file)
        try:
            cat_adf = adfuller(cat_diff, regression='n')  # 移除趋势项，使用简单模型
            dog_adf = adfuller(dog_diff, regression='n')

            print(f"猫数据:", file=output_file)
            print(f"ADF统计量: {cat_adf[0]:.4f}", file=output_file)
            print(f"p值: {cat_adf[1]:.4f}", file=output_file)
            print(f"临界值:", file=output_file)
            for key, value in cat_adf[4].items():
                print(f"\t{key}: {value:.4f}", file=output_file)
            print(
                f"平稳性判断: {'平稳' if cat_adf[1] < 0.05 else '不平稳'}", file=output_file)

            print(f"\n狗数据:", file=output_file)
            print(f"ADF统计量: {dog_adf[0]:.4f}", file=output_file)
            print(f"p值: {dog_adf[1]:.4f}", file=output_file)
            print(f"临界值:", file=output_file)
            for key, value in dog_adf[4].items():
                print(f"\t{key}: {value:.4f}", file=output_file)
            print(
                f"平稳性判断: {'平稳' if dog_adf[1] < 0.05 else '不平稳'}", file=output_file)
        except ValueError as e:
            print(f"ADF检验失败: {str(e)}", file=output_file)

        # 进行白噪声检验
        print(f"\n{country} {i}阶差分的Ljung-Box检验结果：", file=output_file)
        try:
            # 计算最大可用的lag值（不超过数据长度的四分之一）
            max_lag = min(len(cat_diff) // 4, 10)
            if max_lag > 0:  # 确保有足够的数据进行检验
                cat_lb = acorr_ljungbox(cat_diff, lags=range(
                    1, max_lag + 1), return_df=True)
                dog_lb = acorr_ljungbox(dog_diff, lags=range(
                    1, max_lag + 1), return_df=True)

                print(f"猫数据:", file=output_file)
                print(
                    f"统计量: {cat_lb['lb_stat'].iloc[-1]:.4f}", file=output_file)
                print(
                    f"p值: {cat_lb['lb_pvalue'].iloc[-1]:.4f}", file=output_file)
                print(
                    f"白噪声判断: {'是白噪声' if cat_lb['lb_pvalue'].iloc[-1] > 0.05 else '非白噪声'}", file=output_file)

                print(f"\n狗数据:", file=output_file)
                print(
                    f"统计量: {dog_lb['lb_stat'].iloc[-1]:.4f}", file=output_file)
                print(
                    f"p值: {dog_lb['lb_pvalue'].iloc[-1]:.4f}", file=output_file)
                print(
                    f"白噪声判断: {'是白噪声' if dog_lb['lb_pvalue'].iloc[-1] > 0.05 else '非白噪声'}", file=output_file)
            else:
                print("数据量不足，无法进行Ljung-Box检验", file=output_file)
        except Exception as e:
            print(f"Ljung-Box检验失败: {str(e)}", file=output_file)
        print("\n" + "="*50, file=output_file)

# 在所有循环结束后关闭文件
output_file.close()
