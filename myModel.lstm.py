import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置中文字体，防止matplotlib图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 任务一：数据读取、清洗与特征工程
# ==========================================
file_path = 'powerdemand_5min_2021_to_2024_with weather.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到数据文件: {file_path}，请确保文件在当前目录下。")

print("1. 正在读取与清洗数据...")
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df = df.sort_index()

# 缺失值处理：基于时间的线性插值
df = df.interpolate(method='time').bfill().ffill()

# 智能异常值剔除 (同期Z-Score法)
df['time_of_day'] = df.index.strftime('%H:%M')
group_stats = df.groupby('time_of_day')['Power demand'].agg(['mean', 'std']).rename(
    columns={'mean': 'hist_mean', 'std': 'hist_std'}
)
group_stats['hist_std'] = group_stats['hist_std'].replace(0, 1)  # 防止除零
df = df.join(group_stats, on='time_of_day')
df['z_score'] = (df['Power demand'] - df['hist_mean']) / df['hist_std']

anomaly_mask = np.abs(df['z_score']) > 3.5
df.loc[anomaly_mask, 'Power demand'] = np.nan
df['Power demand'] = df['Power demand'].interpolate(method='time')
df.drop(columns=['time_of_day', 'hist_mean', 'hist_std', 'z_score'], inplace=True)

# 特征工程：时间周期编码 (解决时间断层问题)
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

# 筛选建模特征
target_col = 'Power demand'
weather_cols = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']
weather_cols = [c for c in weather_cols if c in df.columns]  # 动态适配列名
time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
features_cols = [target_col] + weather_cols + time_cols

df_model = df[features_cols].copy()

# 数据标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=features_cols, index=df_model.index)
print(f"数据规整完成，最终特征维度: {len(features_cols)}")

# ==========================================
# 任务二：PyTorch 数据集、模型定义与训练
# ==========================================
LOOK_BACK = 288 * 2  # 回顾过去2天 (每天288个点，共576)
FORECAST_HORIZON = 288  # 预测未来24小时
TARGET_IDX = 0  # Power demand 在 features_cols 中的索引
BATCH_SIZE = 64
EPOCHS = 20  # 训练轮数
LR = 0.001


class LoadDataset(Dataset):
    def __init__(self, data, look_back, horizon, target_idx):
        self.data = data
        self.look_back = look_back
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.look_back - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.look_back]
        y = self.data[idx + self.look_back: idx + self.look_back + self.horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最后一个时间步的隐藏状态进行多步预测
        out = self.fc(out[:, -1, :])
        return out


# 划分数据集
data_values = df_scaled.values
split_idx = int(len(data_values) * 0.8)
train_data = data_values[:split_idx]
val_data = data_values[split_idx:]

train_dataset = LoadDataset(train_data, LOOK_BACK, FORECAST_HORIZON, TARGET_IDX)
val_dataset = LoadDataset(val_data, LOOK_BACK, FORECAST_HORIZON, TARGET_IDX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPredictor(input_size=len(features_cols), hidden_size=64, num_layers=2, output_size=FORECAST_HORIZON).to(
    device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\n2. 开始训练模型 (Device: {device})...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}")

# ==========================================
# 预测、反标准化与置信区间计算
# ==========================================
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch).cpu().numpy()
        all_preds.append(y_pred)
        all_trues.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_trues = np.vstack(all_trues)


def inverse_target(pred_scaled, scaler, target_idx, num_features):
    samples, horizon = pred_scaled.shape
    dummy = np.zeros((samples * horizon, num_features))
    dummy[:, target_idx] = pred_scaled.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx].reshape(samples, horizon)


y_pred_actual = inverse_target(all_preds, scaler, TARGET_IDX, len(features_cols))
y_true_actual = inverse_target(all_trues, scaler, TARGET_IDX, len(features_cols))

# 计算每个预测步长的残差标准差 (用于生成动态置信区间)
residuals = y_true_actual - y_pred_actual
std_residuals = np.std(residuals, axis=0)

# 提取最后一个测试样本进行可视化
last_pred = y_pred_actual[-1]
last_true = y_true_actual[-1]
conf_lower = last_pred - 1.96 * std_residuals
conf_upper = last_pred + 1.96 * std_residuals

plt.figure(figsize=(15, 6))
plt.plot(last_true, label='实际负荷', color='blue', linewidth=2)
plt.plot(last_pred, label='预测负荷', color='red', linestyle='--', linewidth=2)
plt.fill_between(range(FORECAST_HORIZON), conf_lower, conf_upper, color='gray', alpha=0.3, label='95% 置信区间')
plt.title('未来24小时区域负荷预测与置信区间 (PyTorch)', fontsize=15)
plt.xlabel('时间步 (5分钟/步)', fontsize=12)
plt.ylabel('电力负荷 (MW)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 任务三：多维异常状态智能识别引擎
# ==========================================
def detect_grid_anomalies(actual_series, pred_series, lower_bound, upper_bound, history_df):
    anomalies = []
    history_df = history_df.copy()
    history_df['time_of_day'] = history_df.index.strftime('%H:%M')

    hist_means = history_df.groupby('time_of_day')['Power demand'].mean()
    hist_diffs = history_df['Power demand'].diff().dropna()
    threshold = hist_diffs.quantile(0.995)  # 动态差分阈值

    for i in range(1, len(actual_series)):
        t = actual_series.index[i]
        actual_val = actual_series.iloc[i]
        pred_val = pred_series[i]
        tod = t.strftime('%H:%M')

        # 【状态1 & 5】突破AI置信区间 / 趋势严重偏离
        if actual_val < lower_bound[i] or actual_val > upper_bound[i]:
            deviation_pct = abs(actual_val - pred_val) / (pred_val + 1e-5)
            risk = '高(红色)' if deviation_pct > 0.15 else '中(橙色)' if deviation_pct > 0.05 else '低(黄色)'
            anomalies.append({'时间': t, '异常类型': '趋势严重偏离(AI残差异常)', '实际负荷': actual_val,
                              '偏离幅度': f"{deviation_pct:.2%}", '风险等级': risk})
            continue

        # 【状态2 & 3】突增 / 突降 (一阶差分突变)
        diff = actual_val - actual_series.iloc[i - 1]
        if abs(diff) > threshold:
            anomalies.append({'时间': t, '异常类型': '负荷突增' if diff > 0 else '负荷突降', '实际负荷': actual_val,
                              '偏离幅度': f"{diff:+.2f} MW", '风险等级': '高(红色)'})
            continue

        # 【状态4】持续偏高 / 持续偏低 (与历史同期均值对比)
        if tod in hist_means.index:
            hist_mean = hist_means[tod]
            pct_diff = (actual_val - hist_mean) / (hist_mean + 1e-5)
            if abs(pct_diff) > 0.20:
                anomalies.append(
                    {'时间': t, '异常类型': '持续偏高' if pct_diff > 0 else '持续偏低', '实际负荷': actual_val,
                     '偏离幅度': f"{pct_diff:+.2%}", '风险等级': '中(橙色)'})

    return pd.DataFrame(anomalies)


# 构造最后24小时的时间索引并运行检测
last_idx = df.index[-FORECAST_HORIZON:]
test_last_actual = pd.Series(last_true, index=last_idx)
test_last_pred = pd.Series(last_pred, index=last_idx)
test_lower = pd.Series(conf_lower, index=last_idx)
test_upper = pd.Series(conf_upper, index=last_idx)

history_df = df.iloc[:split_idx]  # 使用训练集作为历史基准

print("\n3. 正在运行异常状态识别引擎...")
alert_df = detect_grid_anomalies(test_last_actual, test_last_pred.values, test_lower.values, test_upper.values,
                                 history_df)
print(f"共识别出 {len(alert_df)} 个异常状态点：")
if not alert_df.empty:
    print(alert_df.head(10).to_string(index=False))
else:
    print("当前测试窗口内未发现严重异常状态。")