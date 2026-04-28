# -*- coding: utf-8 -*-
"""
Corrected DCC-GARCH model for Bitcoin/Gold safe-haven analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model
from scipy.optimize import minimize

# =========================
# 1. 路径设置
# =========================
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


# =========================
# 2. 读取数据
# =========================
inf_monthly = load_data('data.xlsx', 'Inf_Exp', date_col='Date', is_excel=True, skip=2)
btc = load_data('比特币用这个-CBBTCUSD.csv', 'BTC')
gold = load_data('黄金用这个-gold_lbma_am_usd_2023_2026.csv', 'Gold', date_col='date')
sp500 = load_data('股市-SP500.csv', 'SP500')
vix = load_data('恐慌指标VIXCLS.csv', 'VIX')


# =========================
# 3. 日度频率对齐
# =========================
daily_index = sp500.index.union(btc.index).union(gold.index).sort_values()

# 月度 T5YIFR 前向填充为日度
inf_daily = inf_monthly.reindex(daily_index).ffill()

df = sp500.join([btc, gold, vix, inf_daily], how='left').dropna()


# =========================
# 4. 计算对数收益率
# 注意：GARCH 模型中一般使用百分比收益率，所以保留 *100
# =========================
df['r_BTC'] = np.log(df['BTC'] / df['BTC'].shift(1)) * 100
df['r_Gold'] = np.log(df['Gold'] / df['Gold'].shift(1)) * 100
df['r_SP500'] = np.log(df['SP500'] / df['SP500'].shift(1)) * 100
df['d_Inf'] = df['Inf_Exp'].diff()

df = df.dropna()

print("=" * 60)
print("Data prepared successfully")
print(f"Sample size: {len(df)}")
print(f"Start date: {df.index.min().date()}")
print(f"End date:   {df.index.max().date()}")
print("=" * 60)


# =========================
# 5. 单变量 GARCH(1,1)
# =========================
def fit_garch(series):
    """
    Fit GARCH(1,1) model and return standardized residuals.
    """
    model = arch_model(
        series,
        mean='Constant',
        vol='GARCH',
        p=1,
        q=1,
        dist='t',
        rescale=False
    )

    res = model.fit(disp='off')
    std_resid = res.std_resid.dropna()

    return std_resid, res


std_btc, garch_btc = fit_garch(df['r_BTC'])
std_gold, garch_gold = fit_garch(df['r_Gold'])
std_sp500, garch_sp500 = fit_garch(df['r_SP500'])

print("\nGARCH models fitted successfully.")


# =========================
# 6. 双变量 DCC-GARCH 函数
# =========================
def estimate_dcc(std_resid_1, std_resid_2, name1, name2):
    """
    Estimate bivariate DCC-GARCH dynamic correlations.
    """

    data = pd.concat([std_resid_1, std_resid_2], axis=1).dropna()
    data.columns = [name1, name2]

    eps = data.values
    T = eps.shape[0]

    # 去除极端标准化残差，避免优化不稳定
    eps = np.clip(eps, -8, 8)

    Q_bar = np.cov(eps.T)

    def dcc_neg_loglik(params):
        a, b = params

        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1e10

        Q_t = Q_bar.copy()
        loglik = 0.0

        for t in range(1, T):
            e_lag = eps[t - 1].reshape(-1, 1)

            Q_t = (1 - a - b) * Q_bar + a * (e_lag @ e_lag.T) + b * Q_t

            diag_q = np.sqrt(np.diag(Q_t))
            D_inv = np.diag(1 / diag_q)
            R_t = D_inv @ Q_t @ D_inv

            det_R = np.linalg.det(R_t)

            if det_R <= 0 or np.isnan(det_R):
                return 1e10

            e_t = eps[t].reshape(-1, 1)

            loglik += np.log(det_R) + (e_t.T @ np.linalg.inv(R_t) @ e_t).item()

        return 0.5 * loglik

    # 参数下限避免 DCC 退化成水平线
    bounds = [(0.005, 0.20), (0.70, 0.985)]
    constraints = ({
        'type': 'ineq',
        'fun': lambda x: 0.999 - x[0] - x[1]
    })

    initial_params = np.array([0.05, 0.90])

    result = minimize(
        dcc_neg_loglik,
        initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )

    a, b = result.x

    print("\n" + "=" * 60)
    print(f"DCC result: {name1} vs {name2}")
    print(f"alpha = {a:.4f}")
    print(f"beta  = {b:.4f}")
    print(f"alpha + beta = {a + b:.4f}")
    print(f"Optimization success: {result.success}")
    print("=" * 60)

    # 生成动态相关系数
    Q_t = Q_bar.copy()
    corr_list = []

    for t in range(1, T):
        e_lag = eps[t - 1].reshape(-1, 1)

        Q_t = (1 - a - b) * Q_bar + a * (e_lag @ e_lag.T) + b * Q_t

        diag_q = np.sqrt(np.diag(Q_t))
        D_inv = np.diag(1 / diag_q)
        R_t = D_inv @ Q_t @ D_inv

        corr_list.append(R_t[0, 1])

    corr_series = pd.Series(
        corr_list,
        index=data.index[1:],
        name=f'DCC_{name1}_{name2}'
    )

    return corr_series, a, b


# =========================
# 7. 分别估计 BTC-SP500 和 Gold-SP500
# =========================
dcc_btc_sp500, a_btc, b_btc = estimate_dcc(
    std_btc,
    std_sp500,
    'BTC',
    'SP500'
)

dcc_gold_sp500, a_gold, b_gold = estimate_dcc(
    std_gold,
    std_sp500,
    'Gold',
    'SP500'
)


# =========================
# 8. 合并结果
# =========================
result_df = df.join([dcc_btc_sp500, dcc_gold_sp500], how='inner')

high_stress = result_df[result_df['VIX'] > 18]
normal_period = result_df[result_df['VIX'] <= 18]

print("\n" + "=" * 60)
print("Safe-haven test based on DCC-GARCH")
print(f"High-stress days: {len(high_stress)}")
print("=" * 60)

print("\nAverage DCC correlations:")
print(f"BTC-S&P500 normal period:      {normal_period['DCC_BTC_SP500'].mean():.3f}")
print(f"BTC-S&P500 high-stress period: {high_stress['DCC_BTC_SP500'].mean():.3f}")
print(f"Gold-S&P500 normal period:     {normal_period['DCC_Gold_SP500'].mean():.3f}")
print(f"Gold-S&P500 high-stress period:{high_stress['DCC_Gold_SP500'].mean():.3f}")


# =========================
# 9. 导出 Excel
# =========================
output_file = os.path.join(base_path, 'DCC_GARCH_Dynamic_Correlation_Final.xlsx')
result_df.to_excel(output_file)

print("\nExcel file exported:")
print(output_file)


# =========================
# 10. 画图
# =========================
plt.figure(figsize=(13, 6))

plt.plot(
    result_df.index,
    result_df['DCC_BTC_SP500'],
    label='Bitcoin-S&P 500 DCC',
    linewidth=2
)

plt.plot(
    result_df.index,
    result_df['DCC_Gold_SP500'],
    label='Gold-S&P 500 DCC',
    linewidth=2
)

plt.axhline(0, linestyle='--', linewidth=1)

ymin = min(result_df['DCC_BTC_SP500'].min(), result_df['DCC_Gold_SP500'].min()) - 0.05
ymax = max(result_df['DCC_BTC_SP500'].max(), result_df['DCC_Gold_SP500'].max()) + 0.05

plt.fill_between(
    result_df.index,
    ymin,
    ymax,
    where=result_df['VIX'] > 18,
    alpha=0.15,
    label='High Stress Periods (VIX > 18)'
)

plt.title('DCC-GARCH Dynamic Correlations with S&P 500')
plt.xlabel('Date')
plt.ylabel('Dynamic Conditional Correlation')
plt.ylim(ymin, ymax)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()