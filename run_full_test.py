# -*- coding: utf-8 -*-
"""
完整用户操作流程模拟 - 电力负荷预测与智能分析系统
"""
import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] 已加载 .env 配置文件")
except ImportError:
    print("[INFO] 未安装 python-dotenv，将从环境变量读取配置")

print('=' * 60)
print('⚡ 完整用户操作流程模拟 - 电力负荷预测与智能分析系统')
print('=' * 60)

# ========== 任务1：数据预处理 ==========
print('\n【任务1】负荷数据智能预处理')
print('-' * 40)

# 动态导入 myModel.lstm 模块
import importlib.util
lstm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myModel.lstm.py')
spec = importlib.util.spec_from_file_location("myModel_lstm", lstm_file)
myModel_lstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myModel_lstm)
LoadForecaster = myModel_lstm.LoadForecaster

import pandas as pd

forecaster = LoadForecaster(file_path='powerdemand_5min_2021_to_2024_with weather.csv')
forecaster.load_and_preprocess_data()
print('✅ 数据读取完成')
print(f'   - 特征维度: {len(forecaster.features_cols)}')
print(f'   - 特征列: {forecaster.features_cols}')

# ========== 任务2：模型训练与预测 ==========
print('\n【任务2】短期负荷AI预测')
print('-' * 40)

forecaster.prepare_datasets()
print('✅ 数据集准备完成')

# 训练模型（减少epoch加快速度）
forecaster.EPOCHS = 3  # 减少epoch以加快演示
training_history = forecaster.train_model()
print('✅ 模型训练完成')

# 进行预测
y_pred_actual, y_true_actual, std_residuals = forecaster.predict()
print('✅ 预测计算完成')

# 提取最后一个测试样本
last_pred = y_pred_actual[-1]
last_true = y_true_actual[-1]
forecaster.pred_conf_lower = last_pred - 1.96 * std_residuals
forecaster.pred_conf_upper = last_pred + 1.96 * std_residuals

# 构造时间索引
last_idx = forecaster.df.index[-forecaster.FORECAST_HORIZON:]
test_last_actual = pd.Series(last_true, index=last_idx)
test_last_pred = pd.Series(last_pred, index=last_idx)

# ========== 任务3：异常识别 ==========
print('\n【任务3】负荷异常智能识别')
print('-' * 40)

anomaly_df = forecaster.detect_anomalies(
    test_last_actual,
    test_last_pred.values,
    forecaster.pred_conf_lower,
    forecaster.pred_conf_upper
)
print(f'✅ 异常识别完成，共检测到 {len(anomaly_df)} 个异常')

# ========== 任务4：生成图表 ==========
print('\n【任务4】生成预测可视化图表')
print('-' * 40)

chart_base64 = forecaster.generate_prediction_chart(
    last_true, last_pred,
    forecaster.pred_conf_lower,
    forecaster.pred_conf_upper
)
print(f'✅ 图表生成完成 (Base64长度: {len(chart_base64)}字符)')

# 保存图表
with open('test_prediction_chart.png', 'wb') as f:
    f.write(base64.b64decode(chart_base64))
print('   - 图表已保存: test_prediction_chart.png')

# ========== 统计信息 ==========
stats = forecaster.get_forecast_stats(last_pred, last_true)
print('\n【预测统计信息】')
for key, value in stats.items():
    print(f'   - {key}: {value}')

# ========== 异常列表 ==========
if not anomaly_df.empty:
    print('\n【异常识别结果】')
    for idx, row in anomaly_df.iterrows():
        print(f"   - [{row['风险等级']}] {row['时间']} | {row['异常类型']} | 偏离: {row['偏离幅度']}")
else:
    print('\n【异常识别结果】无异常检测到')

# ========== 任务5：AI简报生成 ==========
print('\n【任务5】预测简报自动生成（调用百度千帆大模型）')
print('-' * 40)

from generate_report import build_analyst_prompt, call_qianfan_agent

# 准备简报数据
stats_for_report = {
    '预测日期': stats['预测日期'],
    '预测时间粒度': stats['预测时间粒度'],
    '预测最大负荷(MW)': stats['预测最大负荷(MW)'],
    '预计峰值出现时间': stats['预计峰值出现时间'],
    '预测最小负荷(MW)': stats['预测最小负荷(MW)'],
    '预计谷值出现时间': stats['预计谷值出现时间'],
    '平均负荷(MW)': stats['平均负荷(MW)'],
    '峰谷差(MW)': stats['峰谷差(MW)'],
    '95%置信区间最大宽度(MW)': stats['95%置信区间最大宽度(MW)'],
    '气象关联特征': '受副热带高压影响，明日气温偏高，局部地区有雷阵雨'
}

prompt = build_analyst_prompt(stats_for_report, anomaly_df)
report = call_qianfan_agent(prompt)

print('\n' + '=' * 60)
print('📝 AI分析简报')
print('=' * 60)
print(report)

# 保存完整报告
report_content = f'''# 电力负荷预测与智能分析系统 - 测试报告

生成时间: 2026-06-28 22:15:00

---

## 一、预测统计信息

| 指标 | 数值 |
|------|------|
| 预测日期 | {stats['预测日期']} |
| 预测时间粒度 | {stats['预测时间粒度']} |
| 预测最大负荷(MW) | {stats['预测最大负荷(MW)']} |
| 预计峰值出现时间 | {stats['预计峰值出现时间']} |
| 预测最小负荷(MW) | {stats['预测最小负荷(MW)']} |
| 预计谷值出现时间 | {stats['预计谷值出现时间']} |
| 平均负荷(MW) | {stats['平均负荷(MW)']} |
| 峰谷差(MW) | {stats['峰谷差(MW)']} |
| 95%置信区间最大宽度(MW) | {stats['95%置信区间最大宽度(MW)']} |

---

## 二、异常识别结果

异常数量: {len(anomaly_df)}

'''

if not anomaly_df.empty:
    report_content += anomaly_df.to_string(index=False)
else:
    report_content += '无异常检测到'

report_content += f'''

---

## 三、AI分析简报

{report}

---

## 四、预测图表

![预测图表](test_prediction_chart.png)

---

*本报告由电力负荷预测与智能分析系统自动生成*
'''

with open('test_full_report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print('\n' + '=' * 60)
print('✅ 测试报告生成完成！')
print('=' * 60)
print('📄 完整报告: test_full_report.md')
print('📊 预测图表: test_prediction_chart.png')