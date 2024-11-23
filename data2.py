# 定义每种宠物的年度食品需求量（千克）
CAT_FOOD_DEMAND = 40  # 猫每年食品需求
DOG_FOOD_DEMAND = 150  # 狗每年食品需求

# 定义地区代表性比例
EUROPE_RATIO = 0.4  # 法国和德国代表欧洲的40%

# 解析数据


def calculate_global_demand(data):
    # 初始化结果字典，按年份存储
    yearly_demands = {2024: 0, 2025: 0, 2026: 0}

    current_country = None
    for line in data.strip().split('\n'):
        if not line:
            continue

        parts = line.split()

        # 如果是年份行，跳过
        if parts[0] == 'year':
            continue

        # 如果是国家名，记录当前国家
        if len(parts) == 1:
            current_country = parts[0]
            continue

        # 处理宠物数据行
        pet_type = parts[0]
        numbers = [int(x) for x in parts[1:]]

        # 计算需求量
        multiplier = CAT_FOOD_DEMAND if pet_type == 'cat' else DOG_FOOD_DEMAND

        # 根据不同地区应用不同系数
        if current_country in ['france', 'germany']:
            # 欧洲数据需要除以40%来得到整个欧洲的数据
            regional_multiplier = 1 / EUROPE_RATIO
        else:
            regional_multiplier = 1

        # 更新每年的需求量
        for year, number in zip([2024, 2025, 2026], numbers):
            yearly_demands[year] += number * multiplier * regional_multiplier

    return yearly_demands


# 使用示例
data = """year 2024 2025 2026

france
cat 1751 1734 1738
dog 1079 1044 1058

germany
cat 1588 1557 1557
dog 1053 1050 1052

america
cat 9138 6920 9029
dog 9463 8290 9040

china
cat 7730 8532 9450
dog 5034 5064 5196"""

demands = calculate_global_demand(data)
for year, demand in demands.items():
    print(f"{year}: {demand:,.0f} kg")
