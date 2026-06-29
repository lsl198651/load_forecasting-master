# -*- coding: utf-8 -*-
"""快速测试脚本 - 验证所有新功能"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["QIANFAN_API_KEY"] = ""

import importlib.util
spec = importlib.util.spec_from_file_location('myModel_lstm', os.path.join(os.path.dirname(__file__), 'myModel.lstm.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
LoadForecaster = mod.LoadForecaster

f = LoadForecaster('powerdemand_5min_2021_to_2024_with weather.csv')
f.EPOCHS = 2

print('=' * 60)
print('功能测试 - 电力负荷预测系统')
print('=' * 60)

print('\n[Step 1/4] 训练24小时模型并保存...')
result = f.run_full_pipeline(forecast_hours=24, save_model=True)
print(f'  ✓ 24h预测完成: 最大负荷={result["stats"]["预测最大负荷(MW)"]} MW')
print(f'  ✓ 模型已保存: {result["model_name"]}')

print('\n[Step 2/4] 列出已保存模型...')
models = f.list_saved_models()
for m in models:
    print(f'  - {m["name"]} ({m["forecast_hours"]}h, {m["epochs"]}轮, val_loss={m.get("final_val_loss", "N/A")})')

print('\n[Step 3/4] 使用已保存模型进行预测...')
result2 = f.run_full_pipeline(forecast_hours=24, use_saved_model=models[0]['name'])
print(f'  ✓ 使用保存模型预测完成: 最大负荷={result2["stats"]["预测最大负荷(MW)"]} MW')
print(f'  ✓ 模型名称: {result2["model_name"]}')

print('\n[Step 4/4] 测试报告导出功能...')
from generate_report import build_analyst_prompt, call_qianfan_agent
import pandas as pd

anomalies_df = pd.DataFrame(result['anomalies'])
prompt = build_analyst_prompt(result['stats'], anomalies_df)
report = call_qianfan_agent(prompt)

report_md, fname_md = f.export_report(result, report, format='md')
print(f'  ✓ Markdown报告已生成: {fname_md} ({len(report_md)} 字符)')

report_html, fname_html = f.export_report(result, report, format='html')
print(f'  ✓ HTML报告已生成: {fname_html} ({len(report_html)} 字符)')

report_txt, fname_txt = f.export_report(result, report, format='txt')
print(f'  ✓ TXT报告已生成: {fname_txt} ({len(report_txt)} 字符)')

print('\n' + '=' * 60)
print('✅ 所有测试通过!')
print('=' * 60)
print('\n测试结果总结:')
print(f'  - 模型训练与保存: ✓ 通过')
print(f'  - 模型列表查询: ✓ 通过')
print(f'  - 模型加载与预测: ✓ 通过')
print(f'  - 报告导出 (md/html/txt): ✓ 通过')
print(f'  - 24小时预测: ✓ 通过')
print(f'  - 已保存模型数: {len(models)}')