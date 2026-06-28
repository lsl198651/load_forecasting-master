import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# 设置中文字体，防止matplotlib图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 配置文件
# ==========================================
class Config:
    # 数据参数
    DATA_PATH = 'powerdemand_5min_2021_to_2024_with weather.csv'

    # 模型参数
    LOOK_BACK = 576  # 回顾过去2天 (每天288个点，共576)
    FORECAST_HORIZON = 288  # 预测未来24小时
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_HEADS = 4
    DROPOUT = 0.2

    # 训练参数
    BATCH_SIZE = 56
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    PATIENCE = 3  # 早停轮数

    # 异常检测参数
    ANOMALY_ZSCORE_THRESHOLD = 3.5
    CONFIDENCE_LEVEL = 1.96  # 95%置信区间

    # 路径参数
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"

    def __init__(self):
        # 动态适配列名
        self.target_col = 'Power demand'
        self.weather_cols = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']

    def get_features(self, df):
        """获取实际存在的特征列"""
        available_weather = [c for c in self.weather_cols if c in df.columns]
        self.time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.features_cols = [self.target_col] + available_weather + self.time_cols
        return self.features_cols


# ==========================================
# 数据预处理模块
# ==========================================
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def load_and_clean(self):
        """加载并清洗数据"""
        print("1. 正在读取与清洗数据...")
        if not os.path.exists(self.config.DATA_PATH):
            raise FileNotFoundError(f"未找到数据文件: {self.config.DATA_PATH}")

        df = pd.read_csv(self.config.DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()

        # 缺失值处理：基于时间的线性插值
        df = df.interpolate(method='time').bfill().ffill()

        # 智能异常值剔除 (同期Z-Score法)
        df = self._remove_anomalies(df)

        # 特征工程：时间周期编码
        df = self._add_time_features(df)

        # 筛选建模特征
        self.config.get_features(df)
        df_model = df[self.config.features_cols].copy()

        # 数据标准化
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_model),
            columns=self.config.features_cols,
            index=df_model.index
        )

        print(f"数据规整完成，最终特征维度: {len(self.config.features_cols)}")
        return df_scaled, df

    def _remove_anomalies(self, df):
        """使用Z-Score方法去除异常值"""
        df['time_of_day'] = df.index.strftime('%H:%M')
        group_stats = df.groupby('time_of_day')['Power demand'].agg(['mean', 'std']).rename(
            columns={'mean': 'hist_mean', 'std': 'hist_std'}
        )
        group_stats['hist_std'] = group_stats['hist_std'].replace(0, 1)
        df = df.join(group_stats, on='time_of_day')
        df['z_score'] = (df['Power demand'] - df['hist_mean']) / df['hist_std']

        anomaly_mask = np.abs(df['z_score']) > self.config.ANOMALY_ZSCORE_THRESHOLD
        df.loc[anomaly_mask, 'Power demand'] = np.nan
        df['Power demand'] = df['Power demand'].interpolate(method='time')
        df.drop(columns=['time_of_day', 'hist_mean', 'hist_std', 'z_score'], inplace=True)

        return df

    def _add_time_features(self, df):
        """添加时间周期编码特征"""
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        return df


# ==========================================
# 数据集类
# ==========================================
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


# ==========================================
# 模型定义
# ==========================================
class TransformerLSTM_PowerDemand(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps, num_heads=4, dropout=0.2):
        super(TransformerLSTM_PowerDemand, self).__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        # LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_steps)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        output = self.fc(pooled)
        return output


# ==========================================
# 训练器类
# ==========================================
class ModelTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_history = {'train_loss': [], 'val_loss': []}

        # 创建保存目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def build_model(self, input_size):
        """构建模型"""
        self.model = TransformerLSTM_PowerDemand(
            input_size=input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            output_steps=self.config.FORECAST_HORIZON,
            num_heads=self.config.NUM_HEADS,
            dropout=self.config.DROPOUT
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        print(f"模型构建完成，设备: {self.device}")

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch + 1:02d}/{self.config.EPOCHS} [训练]',
                  leave=True) as pbar:
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg': f'{train_loss / (batch_idx + 1):.6f}'
                })

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        val_loss = 0

        with tqdm(total=len(val_loader),
                  desc=f'[验证]',
                  leave=True) as pbar:
            with torch.no_grad():
                for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    val_loss += loss.item()

                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'avg': f'{val_loss / (batch_idx + 1):.6f}'
                    })

        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print(f"\n2. 开始训练模型 (Device: {self.device})...")

        for epoch in range(self.config.EPOCHS):
            # 训练
            avg_train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            avg_val_loss = self.validate(val_loader)

            # 记录
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['val_loss'].append(avg_val_loss)

            # 打印结果
            print(f"Epoch {epoch + 1:02d}/{self.config.EPOCHS} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f}")
            print("-" * 60)

            # 保存最佳模型 + 早停
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print(f"✓ 保存最佳模型 (Epoch {epoch + 1}) | 验证损失: {self.best_val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"早停计数: {self.patience_counter}/{self.config.PATIENCE}")

            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n🛑 早停触发！在第 {epoch + 1} 轮停止训练")
                break

        # 保存训练历史
        self.save_history()

        print(f"\n🎉 训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        return self.train_history

    def save_model(self, filename):
        """保存模型"""
        path = os.path.join(self.config.SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_size': self.model.input_proj.in_features,
                'hidden_size': self.config.HIDDEN_SIZE,
                'num_layers': self.config.NUM_LAYERS,
                'output_steps': self.config.FORECAST_HORIZON,
            },
            'best_val_loss': self.best_val_loss
        }, path)

    def load_model(self, filename='best_model.pth'):
        """加载模型"""
        path = os.path.join(self.config.SAVE_DIR, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ 模型加载成功！最佳验证损失: {self.best_val_loss:.6f}")
        return self.model

    def save_history(self):
        """保存训练历史"""
        path = os.path.join(self.config.LOG_DIR, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.train_history, f, indent=2)


# ==========================================
# 评估器类
# ==========================================
class ModelEvaluator:
    def __init__(self, config, scaler, device=None):
        self.config = config
        self.scaler = scaler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = {}

    def predict(self, model, dataloader):
        """执行预测"""
        model.eval()
        all_preds, all_trues = [], []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(dataloader, desc='预测中', leave=True, ncols=100):
                X_batch = X_batch.to(self.device)
                y_pred = model(X_batch).cpu().numpy()
                all_preds.append(y_pred)
                all_trues.append(y_batch.numpy())

        return np.vstack(all_preds), np.vstack(all_trues)

    def inverse_transform(self, scaled_data, target_idx):
        """反标准化"""
        samples, horizon = scaled_data.shape
        inv_result = np.zeros((samples, horizon))
        num_features = len(self.config.features_cols)
        batch_size = 10000

        for i in range(0, samples, batch_size):
            end_idx = min(i + batch_size, samples)
            batch_samples = end_idx - i

            dummy = np.zeros((batch_samples * horizon, num_features))
            dummy[:, target_idx] = scaled_data[i:end_idx].flatten()
            inv_batch = self.scaler.inverse_transform(dummy)[:, target_idx]
            inv_result[i:end_idx] = inv_batch.reshape(batch_samples, horizon)

        return inv_result

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        }
        return metrics

    def evaluate(self, model, dataloader, target_idx=0):
        """完整评估流程"""
        print("\n3. 开始模型评估...")

        # 预测
        all_preds_scaled, all_trues_scaled = self.predict(model, dataloader)

        # 反标准化
        y_pred = self.inverse_transform(all_preds_scaled, target_idx)
        y_true = self.inverse_transform(all_trues_scaled, target_idx)

        # 计算总体指标
        self.metrics = self.calculate_metrics(y_true.flatten(), y_pred.flatten())

        print("\n📊 模型评估结果:")
        print(f"  MAE:  {self.metrics['MAE']:.4f} MW")
        print(f"  RMSE: {self.metrics['RMSE']:.4f} MW")
        print(f"  R²:   {self.metrics['R2']:.4f}")
        print(f"  MAPE: {self.metrics['MAPE']:.2f}%")

        # 计算每个时间步的误差
        self.step_metrics = {
            'MAE_step': np.mean(np.abs(y_true - y_pred), axis=0),
            'RMSE_step': np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        }

        return y_pred, y_true, self.metrics

    def plot_predictions(self, y_true, y_pred, title_prefix="", save_plot=True):
        """可视化预测结果"""
        # 使用最后一个样本
        last_pred = y_pred[-1]
        last_true = y_true[-1]

        # 计算置信区间
        residuals = y_true - y_pred
        std_residuals = np.std(residuals, axis=0)
        conf_lower = last_pred - self.config.CONFIDENCE_LEVEL * std_residuals
        conf_upper = last_pred + self.config.CONFIDENCE_LEVEL * std_residuals

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 子图1：预测vs实际
        ax1 = axes[0]
        ax1.plot(last_true, label='实际负荷', color='blue', linewidth=2)
        ax1.plot(last_pred, label='预测负荷', color='red', linestyle='--', linewidth=2)
        ax1.fill_between(range(self.config.FORECAST_HORIZON), conf_lower, conf_upper,
                         color='gray', alpha=0.3, label='95% 置信区间')
        ax1.set_title(f'{title_prefix}未来24小时区域负荷预测与置信区间', fontsize=15)
        ax1.set_xlabel('时间步 (5分钟/步)', fontsize=12)
        ax1.set_ylabel('电力负荷 (MW)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 子图2：残差分析
        ax2 = axes[1]
        residual = last_true - last_pred
        ax2.plot(residual, color='purple', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.mean(residual), color='red', linestyle='--', label=f'均值: {np.mean(residual):.2f}')
        ax2.axhline(y=np.mean(residual) + 2 * np.std(residual), color='orange', linestyle=':',
                    label=f'±2σ: {2 * np.std(residual):.2f}')
        ax2.axhline(y=np.mean(residual) - 2 * np.std(residual), color='orange', linestyle=':')
        ax2.set_title('预测残差分析', fontsize=15)
        ax2.set_xlabel('时间步 (5分钟/步)', fontsize=12)
        ax2.set_ylabel('残差 (MW)', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{title_prefix}预测结果.png', dpi=300, bbox_inches='tight')

        plt.show()
        return fig

    def save_metrics(self, filename='evaluation_metrics.json'):
        """保存评估指标"""
        os.makedirs('metrics', exist_ok=True)
        path = os.path.join('metrics', filename)

        # 转换numpy类型为Python原生类型
        metrics_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.floating):
                metrics_save[key] = float(value)
            else:
                metrics_save[key] = value

        with open(path, 'w') as f:
            json.dump(metrics_save, f, indent=2)
        print(f"✅ 评估指标已保存至: {path}")


# ==========================================
# 异常检测模块
# ==========================================
class AnomalyDetector:
    def __init__(self, config, history_df):
        self.config = config
        self.history_df = history_df.copy()
        self._prepare_history_stats()

    def _prepare_history_stats(self):
        """准备历史统计数据"""
        self.history_df['time_of_day'] = self.history_df.index.strftime('%H:%M')
        self.hist_means = self.history_df.groupby('time_of_day')['Power demand'].mean()

        hist_diffs = self.history_df['Power demand'].diff().dropna()
        self.threshold = hist_diffs.quantile(0.995)

    def detect_anomalies(self, actual_series, pred_series, lower_bound, upper_bound):
        """检测异常状态"""
        anomalies = []

        for i in range(1, len(actual_series)):
            t = actual_series.index[i]
            actual_val = actual_series.iloc[i]
            pred_val = pred_series[i]
            tod = t.strftime('%H:%M')

            # 状态1: 突破AI置信区间
            if actual_val < lower_bound[i] or actual_val > upper_bound[i]:
                deviation_pct = abs(actual_val - pred_val) / (pred_val + 1e-5)
                risk = '高(红色)' if deviation_pct > 0.15 else '中(橙色)' if deviation_pct > 0.05 else '低(黄色)'
                anomalies.append({
                    '时间': t,
                    '异常类型': '趋势严重偏离(AI残差异常)',
                    '实际负荷': actual_val,
                    '偏离幅度': f"{deviation_pct:.2%}",
                    '风险等级': risk
                })
                continue

            # 状态2&3: 突增/突降
            diff = actual_val - actual_series.iloc[i - 1]
            if abs(diff) > self.threshold:
                anomalies.append({
                    '时间': t,
                    '异常类型': '负荷突增' if diff > 0 else '负荷突降',
                    '实际负荷': actual_val,
                    '偏离幅度': f"{diff:+.2f} MW",
                    '风险等级': '高(红色)'
                })
                continue

            # 状态4: 持续偏高/偏低
            if tod in self.hist_means.index:
                hist_mean = self.hist_means[tod]
                pct_diff = (actual_val - hist_mean) / (hist_mean + 1e-5)
                if abs(pct_diff) > 0.20:
                    anomalies.append({
                        '时间': t,
                        '异常类型': '持续偏高' if pct_diff > 0 else '持续偏低',
                        '实际负荷': actual_val,
                        '偏离幅度': f"{pct_diff:+.2%}",
                        '风险等级': '中(橙色)'
                    })

        return pd.DataFrame(anomalies)

    def detect_and_report(self, y_true, y_pred, conf_lower, conf_upper, original_df, split_idx):
        """执行异常检测并报告"""
        print("\n4. 正在运行异常状态识别引擎...")

        # 构造时间索引
        last_idx = original_df.index[-self.config.FORECAST_HORIZON:]
        actual_series = pd.Series(y_true[-1], index=last_idx)
        pred_series = pd.Series(y_pred[-1], index=last_idx)
        lower_series = pd.Series(conf_lower, index=last_idx)
        upper_series = pd.Series(conf_upper, index=last_idx)

        # 使用训练集作为历史基准
        history_df = original_df.iloc[:split_idx]
        self.history_df = history_df
        self._prepare_history_stats()

        # 检测异常
        alert_df = self.detect_anomalies(
            actual_series,
            pred_series.values,
            lower_series.values,
            upper_series.values
        )

        print(f"共识别出 {len(alert_df)} 个异常状态点：")
        if not alert_df.empty:
            print(alert_df.head(10).to_string(index=False))
            # 保存异常报告
            os.makedirs('reports', exist_ok=True)
            alert_df.to_csv('reports/anomaly_detection_report.csv', index=False, encoding='utf-8-sig')
            print(f"✅ 异常报告已保存至: reports/anomaly_detection_report.csv")
        else:
            print("✅ 当前测试窗口内未发现严重异常")

        return alert_df


# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 配置和预处理
    config = Config()
    preprocessor = DataPreprocessor(config)
    df_scaled, df_original = preprocessor.load_and_clean()

    # 2. 划分数据集
    data_values = df_scaled.values
    split_idx = int(len(data_values) * 0.7)
    train_data = data_values[:split_idx]
    val_data = data_values[split_idx:]

    target_idx = 0  # Power demand 在 features_cols 中的索引

    # 3. 创建数据加载器
    train_dataset = LoadDataset(train_data, config.LOOK_BACK, config.FORECAST_HORIZON, target_idx)
    val_dataset = LoadDataset(val_data, config.LOOK_BACK, config.FORECAST_HORIZON, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(config, device)
    trainer.build_model(input_size=len(config.features_cols))
    trainer.train(train_loader, val_loader)

    # 5. 加载最佳模型
    model = trainer.load_model()

    # 6. 评估模型
    evaluator = ModelEvaluator(config, preprocessor.scaler, device)
    y_pred, y_true, metrics = evaluator.evaluate(model, val_loader, target_idx)

    # 7. 可视化
    evaluator.plot_predictions(y_true, y_pred, title_prefix="PyTorch")

    # 8. 保存指标
    evaluator.save_metrics()

    # 9. 异常检测
    # 计算置信区间
    residuals = y_true - y_pred
    std_residuals = np.std(residuals, axis=0)
    conf_lower = y_pred[-1] - config.CONFIDENCE_LEVEL * std_residuals
    conf_upper = y_pred[-1] + config.CONFIDENCE_LEVEL * std_residuals

    detector = AnomalyDetector(config, df_original)
    alert_df = detector.detect_and_report(
        y_true, y_pred, conf_lower, conf_upper,
        df_original, split_idx
    )

    print("\n✅ 所有任务完成！")


# ==========================================
# 单独评估脚本（可用于加载已训练模型）
# ==========================================
def evaluate_only(model_path='checkpoints/best_model.pth'):
    """仅评估模式 - 加载已训练模型进行评估"""
    print("🔍 启动评估模式...")

    config = Config()
    preprocessor = DataPreprocessor(config)
    df_scaled, df_original = preprocessor.load_and_clean()

    # 划分数据
    data_values = df_scaled.values
    split_idx = int(len(data_values) * 0.7)
    val_data = data_values[split_idx:]

    target_idx = 0
    val_dataset = LoadDataset(val_data, config.LOOK_BACK, config.FORECAST_HORIZON, target_idx)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(config, device)
    trainer.build_model(input_size=len(config.features_cols))

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    trainer.model.load_state_dict(checkpoint)
    trainer.model.to(device)

    # 评估
    evaluator = ModelEvaluator(config, preprocessor.scaler, device)
    y_pred, y_true, metrics = evaluator.evaluate(trainer.model, val_loader, target_idx)

    # 可视化
    evaluator.plot_predictions(y_true, y_pred, title_prefix="PyTorch [评估模式]")

    # 异常检测
    residuals = y_true - y_pred
    std_residuals = np.std(residuals, axis=0)
    conf_lower = y_pred[-1] - config.CONFIDENCE_LEVEL * std_residuals
    conf_upper = y_pred[-1] + config.CONFIDENCE_LEVEL * std_residuals

    detector = AnomalyDetector(config, df_original)
    alert_df = detector.detect_and_report(
        y_true, y_pred, conf_lower, conf_upper,
        df_original, split_idx
    )

    return metrics, alert_df


if __name__ == "__main__":
    # 运行主程序（训练+评估）
    #main()

    # 如果只需要评估，取消下面注释
    evaluate_only('checkpoints/best_model.pth')