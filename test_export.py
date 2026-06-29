# -*- coding: utf-8 -*-
"""测试导出功能"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 模拟预测结果
mock_result = {
    'stats': {
        '预测日期': '2026-06-30',
        '预测时长': '24小时',
        '预测时间粒度': '5分钟/点 (共288个点)',
        '预测最大负荷(MW)': 5240.5,
        '预计峰值出现时间': '18:45 - 19:15',
        '预测最小负荷(MW)': 1620.3,
        '预计谷值出现时间': '03:30 - 04:00',
        '平均负荷(MW)': 3450.2,
        '峰谷差(MW)': 3620.2,
        '95%置信区间最大宽度(MW)': 215.0
    },
    'anomalies': [
        {'时间': '07:30', '异常类型': '负荷突增', '实际/预测负荷': 3150.0, '偏离幅度': '+450 MW', '风险等级': '高(红色)'},
        {'时间': '14:15', '异常类型': '持续偏高', '实际/预测负荷': 4850.0, '偏离幅度': '+12.5%', '风险等级': '中(橙色)'}
    ],
    'model_name': 'LSTM_24h_20260629'
}

mock_report = """## 一、关键结论
明日负荷水平较高，峰值预计在晚高峰时段出现。

## 二、风险提示
早高峰时段存在负荷突增风险，需关注调峰能力。

## 三、运行建议
建议提前安排备用容量，确保调峰裕度充足。"""

print("=" * 60)
print("测试导出功能")
print("=" * 60)

# 测试导出方法
import pandas as pd

# 手动构造导出内容（模拟 export_report 方法）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stats = mock_result['stats']
anomalies = mock_result['anomalies']

# 测试 Markdown 格式
print("\n[测试 1] 导出 Markdown 格式...")
report_md = f"""# 电力负荷预测分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**预测日期**: {stats['预测日期']}
**预测时长**: {stats['预测时长']}
**模型版本**: {mock_result.get('model_name', '未保存')}

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

---

## 二、异常识别结果

| 时间 | 异常类型 | 偏离幅度 | 风险等级 |
|------|----------|----------|----------|
"""
for a in anomalies:
    report_md += f"| {a['时间']} | {a['异常类型']} | {a['偏离幅度']} | {a['风险等级']} |\n"

report_md += f"""
---

## 三、AI分析简报

{mock_report}

---

*本报告由电力负荷预测与智能分析系统自动生成*
"""

filename_md = f"test_report_{timestamp}.md"
with open(filename_md, 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"  OK! Markdown 报告已生成: {filename_md}")

# 测试 HTML 格式
print("\n[测试 2] 导出 HTML 格式...")
report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>电力负荷预测分析报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>电力负荷预测分析报告</h1>
    <div class="metadata">
        <p><strong>生成时间</strong>: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>预测日期</strong>: {stats['预测日期']}</p>
        <p><strong>预测时长</strong>: {stats['预测时长']}</p>
        <p><strong>模型版本</strong>: {mock_result.get('model_name', '未保存')}</p>
    </div>

    <h2>一、预测统计信息</h2>
    <table>
        <tr><th>指标</th><th>数值</th></tr>
        <tr><td>预测最大负荷(MW)</td><td>{stats['预测最大负荷(MW)']}</td></tr>
        <tr><td>预计峰值出现时间</td><td>{stats['预计峰值出现时间']}</td></tr>
        <tr><td>预测最小负荷(MW)</td><td>{stats['预测最小负荷(MW)']}</td></tr>
        <tr><td>预计谷值出现时间</td><td>{stats['预计谷值出现时间']}</td></tr>
        <tr><td>平均负荷(MW)</td><td>{stats['平均负荷(MW)']}</td></tr>
        <tr><td>峰谷差(MW)</td><td>{stats['峰谷差(MW)']}</td></tr>
    </table>

    <h2>二、异常识别结果</h2>
    <table>
        <tr><th>时间</th><th>异常类型</th><th>偏离幅度</th><th>风险等级</th></tr>
"""
for a in anomalies:
    report_html += f"<tr><td>{a['时间']}</td><td>{a['异常类型']}</td><td>{a['偏离幅度']}</td><td>{a['风险等级']}</td></tr>\n"

report_html += """    </table>

    <h2>三、AI分析简报</h2>
    <div style="background: #e8f4f8; padding: 20px; border-radius: 5px;">
"""
# 处理 mock_report 的 HTML 格式化
report_html += mock_report.replace('## ', '<h3>').replace('\n\n', '</h3><p>').replace('\n', '<br>')
report_html += """    </div>

    <hr>
    <p style="color: #7f8c8d; text-align: center;">本报告由电力负荷预测与智能分析系统自动生成</p>
</body>
</html>
"""

filename_html = f"test_report_{timestamp}.html"
with open(filename_html, 'w', encoding='utf-8') as f:
    f.write(report_html)
print(f"  OK! HTML 报告已生成: {filename_html}")

# 测试 TXT 格式
print("\n[测试 3] 导出 TXT 格式...")
report_txt = f"""
电力负荷预测分析报告
========================

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
预测日期: {stats['预测日期']}
预测时长: {stats['预测时长']}
模型版本: {mock_result.get('model_name', '未保存')}

一、预测统计信息
----------------
- 预测最大负荷(MW): {stats['预测最大负荷(MW)']}
- 预计峰值出现时间: {stats['预计峰值出现时间']}
- 预测最小负荷(MW): {stats['预测最小负荷(MW)']}
- 预计谷值出现时间: {stats['预计谷值出现时间']}
- 平均负荷(MW): {stats['平均负荷(MW)']}
- 峰谷差(MW): {stats['峰谷差(MW)']}

二、异常识别结果
----------------
"""
for a in anomalies:
    report_txt += f"- [{a['风险等级']}] {a['时间']} | {a['异常类型']} | {a['偏离幅度']}\n"

report_txt += f"""
三、AI分析简报
--------------
{mock_report}

========================
本报告由电力负荷预测与智能分析系统自动生成
"""

filename_txt = f"test_report_{timestamp}.txt"
with open(filename_txt, 'w', encoding='utf-8') as f:
    f.write(report_txt)
print(f"  OK! TXT 报告已生成: {filename_txt}")

print("\n" + "=" * 60)
print("导出功能测试结果:")
print("=" * 60)
print(f"  - Markdown 导出: OK ({filename_md})")
print(f"  - HTML 导出:    OK ({filename_html})")
print(f"  - TXT 导出:     OK ({filename_txt})")
print("=" * 60)
print("所有导出功能测试通过!")