# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:07:13 2026

@author: zhouguangzhao
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:18:06 2026
优化版：对齐日度频率并生成 Origin 绘图专用表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设置路径
base_path = r'C:\Users\11034\Desktop\留学生作业\data'

def load_data(file_name, col_name, date_col='observation_date', is_excel=False, skip=0):
    path = os.path.join(base_path, file_name)
    if is_excel:
        df = pd.read_excel(path, sheet_name='Sheet1', skiprows=skip)
    else:
        df = pd.read_csv(path, skiprows=skip)
    df.columns = [c.strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return df.rename(columns={df.columns[0]: col_name})

try:
    # 2. 加载数据
    inf_monthly = load_data('data.xlsx', 'Inf_Exp', date_col='Date', is_excel=True, skip=2)
    btc = load_data('比特币用这个-CBBTCUSD.csv', 'BTC')
    gold = load_data('黄金用这个-gold_lbma_am_usd_2023_2026.csv', 'Gold', date_col='date')
    sp500 = load_data('股市-SP500.csv', 'SP500')
    vix = load_data('恐慌指标VIXCLS.csv', 'VIX')

    # ========================================================
    # 3. 核心修改：频率对齐 (与前一段代码逻辑同步)
    # ========================================================
    # 3.1 创建完整的日度时间轴
    daily_index = sp500.index.union(btc.index).union(gold.index).sort_values()
    
    # 3.2 对月度通胀数据进行前向填充
    inf_daily = inf_monthly.reindex(daily_index).ffill()
    
    # 3.3 使用 left join 保持日度样本量
    df = sp500.join([btc, gold, vix, inf_daily], how='left').dropna()

    # 4. 归一化处理（用于走势对比图）
    for col in ['BTC', 'Gold', 'SP500']:
        df[f'Norm_{col}'] = (df[col] / df[col].iloc[0]) * 100

    # 5. 计算收益率（用于后续 Origin 做相关性散点图）
    df['r_BTC'] = np.log(df['BTC'] / df['BTC'].shift(1)) * 100
    df['r_Gold'] = np.log(df['Gold'] / df['Gold'].shift(1)) * 100
    df['r_SP500'] = np.log(df['SP500'] / df['SP500'].shift(1)) * 100
    df['d_Inf'] = df['Inf_Exp'].diff()
    df = df.dropna()

    # 6. 创建画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # --- 子图 1: 累计回报对比 ---
    ax1.plot(df.index, df['Norm_BTC'], label='Bitcoin', color='#f2a900', alpha=0.8)
    ax1.plot(df.index, df['Norm_Gold'], label='Gold', color='#af9500', linewidth=2)
    ax1.plot(df.index, df['Norm_SP500'], label='S&P 500', color='black', linestyle='--')
    ax1.set_title('Normalized Performance (Daily Data, Base=100)')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- 子图 2: 避险分析与 VIX (阈值同步设为 18) ---
    ax2.plot(df.index, df['Norm_SP500'], color='black', label='S&P 500')
    ax2_vix = ax2.twinx()
    ax2_vix.fill_between(df.index, 0, df['VIX'], where=(df['VIX'] > 18), 
                         color='red', alpha=0.15, label='High Stress (VIX > 18)')
    ax2_vix.plot(df.index, df['VIX'], color='red', alpha=0.3, label='VIX Index')
    ax2.set_title('Market Stress Periods (VIX > 18)')
    ax2.legend(loc='upper left')
    ax2_vix.legend(loc='upper right')

    # --- 子图 3: 通胀预期对比 ---
    ax3.plot(df.index, df['Gold'], color='#af9500', label='Gold Price')
    ax3_inf = ax3.twinx()
    ax3_inf.plot(df.index, df['Inf_Exp'], color='green', label='Inflation Expectations (T5YIFR)', linewidth=1.5)
    ax3.set_title('Gold vs. Inflation Expectations (Forward-Filled)')
    ax3.legend(loc='upper left')
    ax3_inf.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

   # ========================================================
    # 7. 导出 3 个独立的 Excel 文件 (方便 Origin 分别导入)
    # ========================================================
    
    # 文件 1：用于画“三线走势对比图”
    path1 = os.path.join(base_path, 'Plot1_Normalized_Line_Chart.xlsx')
    df[['Norm_BTC', 'Norm_Gold', 'Norm_SP500']].to_excel(path1)
    
    # 文件 2：用于画“避险散点图” (仅包含 VIX > 18 的高压日子)
    # 建议在 Origin 中以 r_SP500 为 X 轴，其他两项为 Y 轴做线性拟合
    path2 = os.path.join(base_path, 'Plot2_Safe_Haven_Scatters.xlsx')
    df[df['VIX'] > 18][['r_SP500', 'r_BTC', 'r_Gold']].to_excel(path2)
    
    # 文件 3：用于画“通胀对冲回归图” (全样本)
    # 建议在 Origin 中以 d_Inf 为 X 轴
    path3 = os.path.join(base_path, 'Plot3_Inflation_Hedge_Regression.xlsx')
    df[['d_Inf', 'r_BTC', 'r_Gold']].to_excel(path3)

    print("\n" + "="*50)
    print("✅ Origin 绘图数据已成功分拆导出：")
    print(f"1. 走势图数据: {os.path.basename(path1)}")
    print(f"2. 避险散点数据: {os.path.basename(path2)}")
    print(f"3. 通胀回归数据: {os.path.basename(path3)}")
    print("="*50)

except Exception as e:
    print(f"错误: {e}")