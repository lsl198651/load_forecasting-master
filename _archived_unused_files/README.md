# 已归档的无关文件说明

**创建时间：** 2026-06-29  
**归档目的：** 清理项目目录，将与电力负荷预测系统核心功能无关的文件移至本文件夹保存，而非直接删除。

---

## 文件清单及说明

### 1. `.idea/` - PyCharm IDE配置文件夹
- **原路径：** 项目根目录
- **类型：** IDE配置文件
- **说明：** PyCharm项目的IDE配置文件，包含项目设置、代码检查配置等。这些文件仅用于IDE开发环境，与项目运行无关。
- **是否需要恢复：** ❌ 不需要（如使用PyCharm开发，IDE会自动重新生成）

### 2. `Report/` - 学术报告文件夹
- **原路径：** 项目根目录
- **类型：** 学术报告文档
- **内容：**
  - `Final-Report.pdf` - 课程最终报告
  - `UG-midterm.pptm` - 中期报告PPT
  - `UG_final.pptm` - 最终报告PPT
- **说明：** 这些是学生课程的学术报告文档，与电力负荷预测系统本身无关，属于项目来源背景材料。
- **是否需要恢复：** ⚠️ 可选（如需要查看原始课程报告，可恢复）

### 3. `test.ipynb` - 测试笔记本
- **原路径：** `models/test.ipynb`
- **类型：** Jupyter Notebook测试文件
- **说明：** 开发过程中的临时测试笔记本，用于实验性代码验证。
- **是否需要恢复：** ❌ 不需要

### 4. `delhi.ipynb` - 德里数据测试
- **原路径：** `models/delhi.ipynb`
- **类型：** Jupyter Notebook测试文件
- **说明：** 使用德里地区数据进行预测测试的笔记本，与本项目使用的数据源无关。
- **是否需要恢复：** ❌ 不需要

### 5. `log` - 实验日志文件
- **原路径：** `models/log`
- **类型：** 文本日志文件
- **说明：** 手动记录的模型实验日志，包含各种LSTM/GRU/RNN模型架构测试的RMSE结果。这些是早期开发阶段的实验记录。
- **是否需要恢复：** ❌ 不需要

### 6. `nano.save` - nano编辑器临时文件
- **原路径：** `server/nano.save`
- **类型：** 文本编辑器临时保存文件
- **内容：** 数据库配置模板
- **说明：** nano文本编辑器的临时保存文件，不应存在于正式项目中。数据库配置应使用正式的配置文件。
- **是否需要恢复：** ❌ 不需要

### 7. `forecast_report.md` - 临时报告文件
- **原路径：** 项目根目录
- **类型：** Markdown报告文件
- **说明：** `generate_report.py` 运行时自动生成的临时报告文件。每次运行都会重新生成，不需要保留旧版本。
- **是否需要恢复：** ❌ 不需要（运行程序会自动重新生成）

### 8. `test.py` - 测试Python文件
- **原路径：** `models/test.py`
- **类型：** Python测试脚本
- **内容：** 仅包含一行测试打印代码 `print('lolwa')`
- **说明：** 简单的测试脚本，无实际用途。
- **是否需要恢复：** ❌ 不需要

---

## server文件夹中的归档文件

### 9. `scrap.py` - 数据爬取脚本
- **原路径：** `server/static/scrap.py`
- **类型：** Python数据爬取脚本
- **说明：** 爬取德里SLDC网站数据的脚本，与本项目数据源无关。
- **是否需要恢复：** ❌ 不需要

### 10. `main1.js` - 旧版JavaScript文件
- **原路径：** `server/static/main1.js`
- **类型：** JavaScript文件
- **说明：** 旧版的前端交互脚本，当前使用 `main.js`。
- **是否需要恢复：** ❌ 不需要

### 11. `README.md` - MySQL配置文档
- **原路径：** `server/README.md`
- **类型：** 文档
- **说明：** 描述MySQL数据库配置步骤的文档，本项目使用SQLite数据库。
- **是否需要恢复：** ❌ 不需要

### 12. `db.cnf` - MySQL配置文件
- **原路径：** `server/db.cnf`
- **类型：** 配置文件
- **说明：** MySQL数据库连接配置文件，本项目使用SQLite数据库。
- **是否需要恢复：** ❌ 不需要

---

## models文件夹中的归档文件

### 13. `aws.py` - AWS调度器
- **原路径：** `models/aws.py`
- **类型：** Python调度脚本
- **说明：** 在AWS上每天定时运行模型的调度器。
- **是否需要恢复：** ❌ 不需要

### 14. `aws_arima.py` - AWS ARIMA脚本
- **原路径：** `models/aws_arima.py`
- **类型：** Python脚本
- **说明：** 在AWS上运行ARIMA模型的脚本。
- **是否需要恢复：** ❌ 不需要

### 15. `aws_rnn.py` - AWS RNN脚本
- **原路径：** `models/aws_rnn.py`
- **类型：** Python脚本
- **说明：** 在AWS上运行RNN/LSTM/GRU模型的脚本。
- **是否需要恢复：** ❌ 不需要

### 16. `aws_smoothing.py` - AWS平滑模型脚本
- **原路径：** `models/aws_smoothing.py`
- **类型：** Python脚本
- **说明：** 在AWS上运行SES/SMA/WMA平滑模型的脚本。
- **是否需要恢复：** ❌ 不需要

### 17. `load_scrap.py` - 负荷数据爬取脚本
- **原路径：** `models/load_scrap.py`
- **类型：** Python爬取脚本
- **说明：** 爬取德里SLDC网站负荷数据的脚本。
- **是否需要恢复：** ❌ 不需要

### 18. `whether_scrap.py` - 天气数据爬取脚本
- **原路径：** `models/whether_scrap.py`
- **类型：** Python爬取脚本
- **说明：** 爬取wunderground网站天气数据的脚本。
- **是否需要恢复：** ❌ 不需要

### 19. `pdq_search.py` - ARIMA超参数搜索脚本
- **原路径：** `models/pdq_search.py`
- **类型：** Python脚本
- **说明：** 对ARIMA模型进行网格搜索超参数的脚本。
- **是否需要恢复：** ❌ 不需要

### 20. `readme.md` - 模型说明文档
- **原路径：** `models/readme.md`
- **类型：** 文档
- **说明：** 描述项目中实现的各种模型的说明文档。
- **是否需要恢复：** ⚠️ 可选（如需要查看模型历史说明）

---

## 如何恢复文件

如果您需要恢复某个文件到原位置，可以执行以下操作：

```powershell
# 恢复单个文件示例（以Report文件夹为例）
Move-Item -Path "_archived_unused_files\Report" -Destination "..\Report" -Force

# 恢复所有文件
Move-Item -Path "_archived_unused_files\*" -Destination ".." -Force
```

---

## 项目核心文件保留清单

以下文件是项目的核心组成部分，已保留在项目根目录：

- `generate_report.py` - 报告生成核心功能
- `myModel.lstm.py` - LSTM模型训练与预测核心
- `test_features.py` - 功能快速测试脚本
- `run_full_test.py` - 完整流程测试脚本
- `start_server.bat` - Web服务器启动脚本
- `celery.sh` - Celery异步任务启动脚本
- `系统使用指南.md` - 项目使用说明文档
- `README.md` - 项目介绍文档
- `LICENSE` - 开源许可证
- `.gitignore` - Git版本控制忽略配置
- `server/` - Django Web服务器（已移除nano.save）
- `models/` - 模型实验代码（已移除test.ipynb等无关文件）
- `saved_models/` - 已训练保存的模型权重
- `screenshots/` - Web界面截图

---

**备注：** 本归档文件夹仅用于保存非核心文件，方便项目管理和清理。所有文件均已妥善保存，可根据需要随时恢复。