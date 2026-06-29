import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from datetime import datetime, timedelta
from io import BytesIO
import base64
import pickle
import joblib

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LoadForecaster:
    """电力负荷预测系统 - 支持模型持久化与多时长预测"""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    def __init__(self, file_path='powerdemand_5min_2021_to_2024_with weather.csv'):
        self.file_path = file_path
        self.df = None
        self.df_model = None
        self.scaler = None
        self.df_scaled = None
        self.model = None
        self.features_cols = None
        self.train_loader = None
        self.val_loader = None
        self.pred_conf_lower = None
        self.pred_conf_upper = None
        self.split_idx = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.LOOK_BACK = 288 * 2
        self.FORECAST_HORIZON = 288
        self.TARGET_IDX = 0
        self.BATCH_SIZE = 64
        self.EPOCHS = 20
        self.LR = 0.001

        self.training_history = None
        self.last_prediction_result = None

        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def set_forecast_horizon(self, hours):
        """设置预测时长（24小时或72小时）"""
        if hours not in [24, 72]:
            raise ValueError("仅支持24小时或72小时预测")
        self.FORECAST_HORIZON = 288 * (hours // 24)
        self.LOOK_BACK = 288 * 2 * (hours // 24)
        return self.FORECAST_HORIZON

    def get_forecast_hours(self):
        """获取当前预测时长（小时）"""
        return self.FORECAST_HORIZON * 5 // 60

    def load_and_preprocess_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"未找到数据文件: {self.file_path}")

        print("正在读取与清洗数据...")
        df = pd.read_csv(self.file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()

        df = df.interpolate(method='time').bfill().ffill()

        df['time_of_day'] = df.index.strftime('%H:%M')
        group_stats = df.groupby('time_of_day')['Power demand'].agg(['mean', 'std']).rename(
            columns={'mean': 'hist_mean', 'std': 'hist_std'}
        )
        group_stats['hist_std'] = group_stats['hist_std'].replace(0, 1)
        df = df.join(group_stats, on='time_of_day')
        df['z_score'] = (df['Power demand'] - df['hist_mean']) / df['hist_std']

        anomaly_mask = np.abs(df['z_score']) > 3.5
        df.loc[anomaly_mask, 'Power demand'] = np.nan
        df['Power demand'] = df['Power demand'].interpolate(method='time')
        df.drop(columns=['time_of_day', 'hist_mean', 'hist_std', 'z_score'], inplace=True)

        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        target_col = 'Power demand'
        weather_cols = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']
        weather_cols = [c for c in weather_cols if c in df.columns]
        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        self.features_cols = [target_col] + weather_cols + time_cols

        self.df_model = df[self.features_cols].copy()

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df_model),
            columns=self.features_cols,
            index=self.df_model.index
        )

        self.df = df
        print(f"数据规整完成，最终特征维度: {len(self.features_cols)}")
        return self.df_model

    def prepare_datasets(self):
        data_values = self.df_scaled.values
        split_idx = int(len(data_values) * 0.8)
        train_data = data_values[:split_idx]
        val_data = data_values[split_idx:]

        train_dataset = LoadDataset(train_data, self.LOOK_BACK, self.FORECAST_HORIZON, self.TARGET_IDX)
        val_dataset = LoadDataset(val_data, self.LOOK_BACK, self.FORECAST_HORIZON, self.TARGET_IDX)

        self.train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        self.split_idx = split_idx
        return train_dataset, val_dataset

    def train_model(self):
        if self.train_loader is None:
            self.prepare_datasets()

        self.model = LSTMPredictor(
            input_size=len(self.features_cols),
            hidden_size=64,
            num_layers=3,
            output_size=self.FORECAST_HORIZON
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)

        print(f"开始训练模型 (Device: {self.device})...")
        training_history = []

        for epoch in range(self.EPOCHS):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            epoch_loss = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(self.train_loader),
                'val_loss': val_loss / len(self.val_loader)
            }
            training_history.append(epoch_loss)
            print(f"Epoch {epoch + 1:02d}/{self.EPOCHS} | Train Loss: {epoch_loss['train_loss']:.6f} | Val Loss: {epoch_loss['val_loss']:.6f}")

        self.training_history = training_history
        return training_history

    def save_model(self, model_name=None):
        """保存模型及相关参数到磁盘"""
        if self.model is None:
            raise ValueError("没有可保存的模型，请先训练模型")

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_model_{timestamp}"

        model_path = os.path.join(self.MODEL_DIR, model_name)
        os.makedirs(model_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(model_path, 'model_weights.pth'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))

        config = {
            'features_cols': self.features_cols,
            'LOOK_BACK': self.LOOK_BACK,
            'FORECAST_HORIZON': self.FORECAST_HORIZON,
            'TARGET_IDX': self.TARGET_IDX,
            'input_size': len(self.features_cols),
            'hidden_size': 64,
            'num_layers': 3,
            'output_size': self.FORECAST_HORIZON,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_history': self.training_history,
            'model_name': model_name,
            'forecast_hours': self.get_forecast_hours()
        }

        with open(os.path.join(model_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"模型已保存至: {model_path}")
        return model_name

    def load_model(self, model_name):
        """从磁盘加载模型及相关参数"""
        model_path = os.path.join(self.MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.features_cols = config['features_cols']
        self.LOOK_BACK = config['LOOK_BACK']
        self.FORECAST_HORIZON = config['FORECAST_HORIZON']
        self.TARGET_IDX = config['TARGET_IDX']
        self.training_history = config.get('training_history', [])

        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))

        self.model = LSTMPredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size']
        ).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model_weights.pth'), map_location=self.device))
        self.model.eval()

        print(f"模型已加载: {model_name} (创建时间: {config.get('created_at', '未知')})")
        return config

    def list_saved_models(self):
        """列出所有已保存的模型"""
        models = []
        if not os.path.exists(self.MODEL_DIR):
            return models

        for name in os.listdir(self.MODEL_DIR):
            model_path = os.path.join(self.MODEL_DIR, name)
            config_path = os.path.join(model_path, 'config.json')
            if os.path.isdir(model_path) and os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    models.append({
                        'name': name,
                        'created_at': config.get('created_at', '未知'),
                        'forecast_hours': config.get('forecast_hours', 24),
                        'epochs': len(config.get('training_history', [])),
                        'final_val_loss': config.get('training_history', [{}])[-1].get('val_loss', None) if config.get('training_history') else None
                    })
                except Exception:
                    pass
        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    def predict(self):
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        if self.val_loader is None:
            self.prepare_datasets()

        self.model.eval()
        all_preds, all_trues = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
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

        y_pred_actual = inverse_target(all_preds, self.scaler, self.TARGET_IDX, len(self.features_cols))
        y_true_actual = inverse_target(all_trues, self.scaler, self.TARGET_IDX, len(self.features_cols))

        residuals = y_true_actual - y_pred_actual
        std_residuals = np.std(residuals, axis=0)

        return y_pred_actual, y_true_actual, std_residuals

    def detect_anomalies(self, actual_series, pred_series, lower_bound, upper_bound, history_df=None):
        anomalies = []

        if history_df is None:
            history_df = self.df.iloc[:self.split_idx]

        history_df = history_df.copy()
        history_df['time_of_day'] = history_df.index.strftime('%H:%M')
        hist_means = history_df.groupby('time_of_day')['Power demand'].mean()
        hist_diffs = history_df['Power demand'].diff().dropna()
        threshold = hist_diffs.quantile(0.995)

        for i in range(1, len(actual_series)):
            t = actual_series.index[i]
            actual_val = actual_series.iloc[i]
            pred_val = pred_series[i]
            tod = t.strftime('%H:%M')

            if actual_val < lower_bound[i] or actual_val > upper_bound[i]:
                deviation_pct = abs(actual_val - pred_val) / (pred_val + 1e-5)
                risk = '高(红色)' if deviation_pct > 0.15 else '中(橙色)' if deviation_pct > 0.05 else '低(黄色)'
                anomalies.append({
                    '时间': t.strftime('%Y-%m-%d %H:%M'),
                    '异常类型': '趋势严重偏离(AI残差异常)',
                    '实际负荷': round(actual_val, 2),
                    '预测负荷': round(pred_val, 2),
                    '偏离幅度': f"{deviation_pct:.2%}",
                    '风险等级': risk
                })
                continue

            diff = actual_val - actual_series.iloc[i - 1]
            if abs(diff) > threshold:
                anomalies.append({
                    '时间': t.strftime('%Y-%m-%d %H:%M'),
                    '异常类型': '负荷突增' if diff > 0 else '负荷突降',
                    '实际负荷': round(actual_val, 2),
                    '预测负荷': round(pred_val, 2),
                    '偏离幅度': f"{diff:+.2f} MW",
                    '风险等级': '高(红色)'
                })
                continue

            if tod in hist_means.index:
                hist_mean = hist_means[tod]
                pct_diff = (actual_val - hist_mean) / (hist_mean + 1e-5)
                if abs(pct_diff) > 0.20:
                    anomalies.append({
                        '时间': t.strftime('%Y-%m-%d %H:%M'),
                        '异常类型': '持续偏高' if pct_diff > 0 else '持续偏低',
                        '实际负荷': round(actual_val, 2),
                        '预测负荷': round(pred_val, 2),
                        '偏离幅度': f"{pct_diff:+.2%}",
                        '风险等级': '中(橙色)'
                    })

        return pd.DataFrame(anomalies)

    def generate_prediction_chart(self, last_true, last_pred, conf_lower, conf_upper):
        hours = self.get_forecast_hours()
        fig, ax = plt.subplots(figsize=(16, 7))

        x_labels = []
        for i in range(0, self.FORECAST_HORIZON, max(1, self.FORECAST_HORIZON // 12)):
            hour = i * 5 // 60
            minute = i * 5 % 60
            x_labels.append((i, f"{hour:02d}:{minute:02d}"))

        ax.plot(last_true, label='实际负荷', color='#1976d2', linewidth=2.5, alpha=0.9)
        ax.plot(last_pred, label='预测负荷', color='#f44336', linewidth=2.5, linestyle='--', alpha=0.9)
        ax.fill_between(range(self.FORECAST_HORIZON), conf_lower, conf_upper,
                       color='#90caf9', alpha=0.3, label='95% 置信区间')

        peak_idx = np.argmax(last_pred)
        valley_idx = np.argmin(last_pred)
        ax.scatter([peak_idx], [last_pred[peak_idx]], color='#d32f2f', s=120, zorder=5, label=f'峰值 {last_pred[peak_idx]:.1f} MW')
        ax.scatter([valley_idx], [last_pred[valley_idx]], color='#388e3c', s=120, zorder=5, label=f'谷值 {last_pred[valley_idx]:.1f} MW')

        ax.set_title(f'未来{hours}小时区域负荷预测与置信区间', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('电力负荷 (MW)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')

        ax.set_xticks([x[0] for x in x_labels])
        ax.set_xticklabels([x[1] for x in x_labels], rotation=45)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        return img_base64

    def get_forecast_stats(self, last_pred, last_true):
        max_load = np.max(last_pred)
        min_load = np.min(last_pred)
        avg_load = np.mean(last_pred)
        peak_diff = max_load - min_load

        peak_idx = np.argmax(last_pred)
        valley_idx = np.argmin(last_pred)

        peak_time = f"{peak_idx * 5 // 60:02d}:{peak_idx * 5 % 60:02d}"
        valley_time = f"{valley_idx * 5 // 60:02d}:{valley_idx * 5 % 60:02d}"

        conf_width = np.max(self.pred_conf_upper - self.pred_conf_lower)
        hours = self.get_forecast_hours()

        stats = {
            '预测日期': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            '预测时长': f"{hours}小时",
            '预测时间粒度': f"5分钟/点 (共{self.FORECAST_HORIZON}个点)",
            '预测最大负荷(MW)': round(max_load, 2),
            '预计峰值出现时间': peak_time,
            '预测最小负荷(MW)': round(min_load, 2),
            '预计谷值出现时间': valley_time,
            '平均负荷(MW)': round(avg_load, 2),
            '峰谷差(MW)': round(peak_diff, 2),
            '95%置信区间最大宽度(MW)': round(conf_width, 2),
            '气象关联特征': '基于历史气象数据训练的模型预测'
        }

        return stats

    def run_full_pipeline(self, forecast_hours=24, use_saved_model=None, save_model=True):
        """
        运行完整的预测流程

        Args:
            forecast_hours: 预测时长，24或72
            use_saved_model: 使用已保存的模型名称，None表示重新训练
            save_model: 是否保存训练后的模型

        Returns:
            预测结果字典
        """
        self.set_forecast_horizon(forecast_hours)

        if self.df is None:
            self.load_and_preprocess_data()

        self.prepare_datasets()

        if use_saved_model:
            print(f"加载已保存模型: {use_saved_model}")
            config = self.load_model(use_saved_model)
            training_history = self.training_history
            model_name = use_saved_model
        else:
            training_history = self.train_model()
            model_name = self.save_model() if save_model else None

        y_pred_actual, y_true_actual, std_residuals = self.predict()

        last_pred = y_pred_actual[-1]
        last_true = y_true_actual[-1]
        self.pred_conf_lower = last_pred - 1.96 * std_residuals
        self.pred_conf_upper = last_pred + 1.96 * std_residuals

        last_idx = self.df.index[-self.FORECAST_HORIZON:]
        test_last_actual = pd.Series(last_true, index=last_idx)
        test_last_pred = pd.Series(last_pred, index=last_idx)

        anomaly_df = self.detect_anomalies(
            test_last_actual,
            test_last_pred.values,
            self.pred_conf_lower,
            self.pred_conf_upper
        )

        chart_base64 = self.generate_prediction_chart(
            last_true, last_pred, self.pred_conf_lower, self.pred_conf_upper
        )

        stats = self.get_forecast_stats(last_pred, last_true)

        prediction_data = {
            'timestamps': [t.strftime('%Y-%m-%d %H:%M') for t in last_idx],
            'actual_values': last_true.tolist(),
            'predicted_values': last_pred.tolist(),
            'confidence_lower': self.pred_conf_lower.tolist(),
            'confidence_upper': self.pred_conf_upper.tolist()
        }

        result = {
            'training_history': training_history,
            'prediction_data': prediction_data,
            'anomalies': anomaly_df.to_dict('records') if not anomaly_df.empty else [],
            'chart_base64': chart_base64,
            'stats': stats,
            'model_name': model_name,
            'forecast_hours': forecast_hours
        }

        self.last_prediction_result = result
        return result

    def export_report(self, result, report_content, format='md'):
        """
        导出分析报告

        Args:
            result: 预测结果字典
            report_content: AI生成的报告内容
            format: 导出格式 (md, html, txt)

        Returns:
            报告内容（字符串）和文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_forecast_report_{timestamp}.{format}"

        stats = result['stats']
        anomalies = result['anomalies']

        if format == 'md':
            report_md = f"""# 电力负荷预测分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**预测日期**: {stats['预测日期']}
**预测时长**: {stats['预测时长']}
**模型版本**: {result.get('model_name', '未保存')}

---

## 一、预测统计信息

| 指标 | 数值 |
|------|------|
| 预测最大负荷(MW) | {stats['预测最大负荷(MW)']} |
| 预计峰值出现时间 | {stats['预计峰值出现时间']} |
| 预测最小负荷(MW) | {stats['预测最小负荷(MW)']} |
| 预计谷值出现时间 | {stats['预计谷值出现时间']} |
| 平均负荷(MW) | {stats['平均负荷(MW)']} |
| 峰谷差(MW) | {stats['峰谷差(MW)']} |
| 95%置信区间最大宽度(MW) | {stats['95%置信区间最大宽度(MW)']} |

---

## 二、异常识别结果

**异常数量**: {len(anomalies)}

"""
            if anomalies:
                report_md += "| 时间 | 异常类型 | 实际负荷(MW) | 预测负荷(MW) | 偏离幅度 | 风险等级 |\n"
                report_md += "|------|----------|-------------|-------------|----------|----------|\n"
                for a in anomalies:
                    report_md += f"| {a['时间']} | {a['异常类型']} | {a['实际负荷']} | {a['预测负荷']} | {a['偏离幅度']} | {a['风险等级']} |\n"
            else:
                report_md += "未检测到异常状态。\n"

            report_md += f"""
---

## 三、AI智能分析

{report_content}

---

## 四、预测图表

![预测图表](data:image/png;base64,{result['chart_base64']})

---

*本报告由电力负荷预测与智能分析系统自动生成*
"""
            return report_md, filename

        elif format == 'txt':
            txt = f"电力负荷预测分析报告\n"
            txt += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            txt += f"预测日期: {stats['预测日期']}\n"
            txt += f"预测时长: {stats['预测时长']}\n"
            txt += "=" * 50 + "\n\n"
            txt += "一、预测统计信息\n"
            txt += "-" * 30 + "\n"
            for k, v in stats.items():
                txt += f"{k}: {v}\n"
            txt += f"\n二、异常识别结果 (共{len(anomalies)}个)\n"
            txt += "-" * 30 + "\n"
            if anomalies:
                for a in anomalies[:10]:
                    txt += f"[{a['风险等级']}] {a['时间']} {a['异常类型']} 偏离: {a['偏离幅度']}\n"
            else:
                txt += "未检测到异常状态\n"
            txt += "\n三、AI智能分析\n"
            txt += "-" * 30 + "\n"
            txt += report_content
            return txt, filename

        elif format == 'html':
            report_html = report_content.replace('\n', '<br>').replace('## ', '<h3>').replace('### ', '<h4>')
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>电力负荷预测分析报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; padding: 40px; background: #f5f7fa; }}
        .report {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #455a64; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f5f5f5; color: #455a64; }}
        .high-risk {{ background: #ffe6e6; color: #d32f2f; }}
        .medium-risk {{ background: #fff4e6; color: #f57c00; }}
        .low-risk {{ background: #fff9e6; color: #f9a825; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 20px 0; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
        .analysis {{ background: #f3e5f5; padding: 20px; border-radius: 8px; line-height: 1.8; }}
    </style>
</head>
<body>
<div class="report">
    <h1>电力负荷预测分析报告</h1>
    <div class="meta">
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>预测日期:</strong> {stats['预测日期']} | <strong>预测时长:</strong> {stats['预测时长']}</p>
        <p><strong>模型版本:</strong> {result.get('model_name', '未保存')}</p>
    </div>

    <h2>一、预测统计信息</h2>
    <table>
        <tr><th>指标</th><th>数值</th></tr>
        <tr><td>预测最大负荷(MW)</td><td><strong>{stats['预测最大负荷(MW)']}</strong></td></tr>
        <tr><td>预计峰值出现时间</td><td>{stats['预计峰值出现时间']}</td></tr>
        <tr><td>预测最小负荷(MW)</td><td><strong>{stats['预测最小负荷(MW)']}</strong></td></tr>
        <tr><td>预计谷值出现时间</td><td>{stats['预计谷值出现时间']}</td></tr>
        <tr><td>平均负荷(MW)</td><td>{stats['平均负荷(MW)']}</td></tr>
        <tr><td>峰谷差(MW)</td><td>{stats['峰谷差(MW)']}</td></tr>
        <tr><td>95%置信区间最大宽度(MW)</td><td>{stats['95%置信区间最大宽度(MW)']}</td></tr>
    </table>

    <h2>二、异常识别结果</h2>
    <p><strong>异常数量: {len(anomalies)}</strong></p>
"""
            if anomalies:
                html += "    <table>\n        <tr><th>时间</th><th>异常类型</th><th>实际负荷(MW)</th><th>预测负荷(MW)</th><th>偏离幅度</th><th>风险等级</th></tr>\n"
                for a in anomalies:
                    risk_class = 'high-risk' if '高' in a['风险等级'] else 'medium-risk' if '中' in a['风险等级'] else 'low-risk'
                    html += f"        <tr class=\"{risk_class}\"><td>{a['时间']}</td><td>{a['异常类型']}</td><td>{a['实际负荷']}</td><td>{a['预测负荷']}</td><td>{a['偏离幅度']}</td><td>{a['风险等级']}</td></tr>\n"
                html += "    </table>\n"
            else:
                html += "    <p style='color: #4caf50;'>✓ 未检测到异常状态</p>\n"

            html += f"""
    <h2>三、AI智能分析</h2>
    <div class="analysis">
        {report_html}
    </div>

    <h2>四、预测图表</h2>
    <img src="data:image/png;base64,{result['chart_base64']}" alt="预测图表">

    <hr style="margin-top: 40px; border: none; border-top: 1px solid #e0e0e0;">
    <p style="text-align: center; color: #999; font-size: 12px;">本报告由电力负荷预测与智能分析系统自动生成</p>
</div>
</body>
</html>"""
            return html, filename

        return report_content, filename


if __name__ == "__main__":
    forecaster = LoadForecaster()

    print("已保存模型列表:")
    saved_models = forecaster.list_saved_models()
    if saved_models:
        for m in saved_models:
            print(f"  - {m['name']} (创建于 {m['created_at']}, {m['forecast_hours']}小时)")
    else:
        print("  暂无已保存的模型")

    print("\n开始新的24小时预测...")
    result = forecaster.run_full_pipeline(forecast_hours=24, save_model=True)

    print("\n" + "=" * 50)
    print("预测结果摘要:")
    print("=" * 50)
    print(json.dumps(result['stats'], ensure_ascii=False, indent=2))
    print(f"\n异常识别数量: {len(result['anomalies'])}")
    print(f"模型名称: {result['model_name']}")

    print("\n已保存模型列表（更新后）:")
    saved_models = forecaster.list_saved_models()
    for m in saved_models:
        print(f"  - {m['name']} (创建于 {m['created_at']}, {m['forecast_hours']}小时)")
