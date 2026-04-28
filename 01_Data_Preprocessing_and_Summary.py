# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:05:25 2026

@author: zhouguangzhao
"""

import pandas as pd
import numpy as np
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
    # 2. 加载原始数据
    inf_monthly = load_data('data.xlsx', 'Inf_Exp', date_col='Date', is_excel=True, skip=2)
    btc = load_data('比特币用这个-CBBTCUSD.csv', 'BTC')
    gold = load_data('黄金用这个-gold_lbma_am_usd_2023_2026.csv', 'Gold', date_col='date')
    sp500 = load_data('股市-SP500.csv', 'SP500')
    vix = load_data('恐慌指标VIXCLS.csv', 'VIX')

    # ========================================================
    # 3. 核心修改：频率对齐 (Per Manuscript Section 3.2)
    # ========================================================
    
    # 3.1 以资产价格的日度日期为准，创建一个完整的日期轴
    # 避免 inner join 导致的数据大规模丢失
    daily_index = sp500.index.union(btc.index).union(gold.index).sort_values()
    
    # 3.2 对月度通胀数据进行“前向填充” (Forward-Filling)
    # 这样每一天都有了对应的市场通胀预期水平
    inf_daily = inf_monthly.reindex(daily_index).ffill()
    
    # 3.3 合并所有日度数据
    # 现在合并后的 df 依然保持日度频率，样本量充足
    df = sp500.join([btc, gold, vix, inf_daily], how='left').dropna()

    # 4. 计算收益率与变动率
    df['r_BTC'] = np.log(df['BTC'] / df['BTC'].shift(1)) * 100
    df['r_Gold'] = np.log(df['Gold'] / df['Gold'].shift(1)) * 100
    df['r_SP500'] = np.log(df['SP500'] / df['SP500'].shift(1)) * 100
    
    # 根据 3.2 章节要求，对通胀预期取“一阶差分”以保持平稳
    df['d_Inf'] = df['Inf_Exp'].diff() 
    df = df.dropna()

    print("\n" + "="*50)
    print("📊 频率对齐后（Forward-Filled）的实证数据")
    print(f"   总观测天数: {len(df)} 天 (已保留日度高频特征)")
    print("="*50)

    # 5. 稳健性避险分析 (VIX > 18)
    vix_threshold = 18 
    high_vix = df[df['VIX'] > vix_threshold]
    
    # 如果 VIX 样本还是少，就按股市大跌日分析
    if len(high_vix) < 10:
        print(f"(VIX > {vix_threshold} 样本较少，分析结果辅以股市大跌日验证)")
        high_vix = df[df['r_SP500'] < -1.5]
    
    safe_haven_corr = high_vix[['r_BTC', 'r_Gold', 'r_SP500']].corr()['r_SP500']
    inf_hedge_corr = df[['r_BTC', 'r_Gold', 'd_Inf']].corr()['d_Inf']

    print(f"\n[1] 避险测试 (压力样本量: {len(high_vix)} 天):")
    print(f"    - 比特币与股市相关性: {safe_haven_corr['r_BTC']:.3f}")
    print(f"    - 黄金与股市相关性:   {safe_haven_corr['r_Gold']:.3f}")

    print(f"\n[2] 通胀对冲测试 (全样本日度差分):")
    print(f"    - 比特币 vs 通胀预期: {inf_hedge_corr['r_BTC']:.3f}")
    print(f"    - 黄金 vs 通胀预期:   {inf_hedge_corr['r_Gold']:.3f}")

except Exception as e:
    print(f"运行出错: {e}")