# 电力负荷预测与智能分析系统

基于深度学习的电力负荷预测平台，集成 AI 大模型自动生成分析报告，提供可视化仪表板和多格式报告导出功能。

## 🚀 功能特性

- **⚡ LSTM 深度学习预测**：采用 PyTorch LSTM 模型，支持 24/72 小时负荷预测，精度高、响应快
- **🔍 智能异常检测**：基于 Z-score 统计方法，实时识别负荷异常，预警调度风险
- **🤖 AI 智能报告生成**：支持百度千帆、阿里通义千问、字节豆包、科大讯飞星火等多种大模型
- **📊 可视化仪表板**：现代化 Web 界面，实时图表展示，交互式数据分析体验
- **📥 多格式报告导出**：支持 Markdown、HTML、TXT 三种格式导出
- **💾 模型持久化**：训练好的模型自动保存，支持复用和加载
- **🔐 安全配置**：通过 `.env` 文件管理 API Key，敏感信息不提交代码仓库

## 🛠️ 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **前端** | Bootstrap 5 + Chart.js | 响应式界面，交互式图表 |
| **后端** | Django 4.x + Django REST Framework | RESTful API 服务 |
| **算法** | PyTorch LSTM | 深度学习负荷预测模型 |
| **大模型** | 百度千帆 / 阿里通义千问 / 字节豆包 / 科大讯飞星火 | AI 报告生成 |
| **数据处理** | Pandas + NumPy + scikit-learn | 数据预处理与特征工程 |
| **数据库** | SQLite | 轻量级数据库，无需额外安装 |

## 📦 快速开始

### 环境要求

- Python 3.8+
- pip 包管理工具

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/lsl198651/load_forecasting-master.git
cd load_forecasting-master
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

复制 `.env` 文件并配置 API Key：

```bash
# .env 文件配置示例
QIANFAN_API_KEY=your_qianfan_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DOUBAN_API_KEY=your_douban_api_key_here
XUNFEI_API_KEY=your_xunfei_api_key_here
```

4. **启动服务**

```bash
# 方式一：双击运行（推荐）
start_server.bat

# 方式二：命令行运行
python start_server.py

# 方式三：直接启动 Django
cd server
python manage.py runserver 127.0.0.1:8000
```

5. **访问系统**

启动后自动打开浏览器，访问：

- **欢迎页**：http://127.0.0.1:8000/
- **预测仪表板**：http://127.0.0.1:8000/dashboard/
- **管理后台**：http://127.0.0.1:8000/admin/

## 📖 使用说明

### 预测流程

1. **选择预测时长**：在仪表板中选择 24 小时或 72 小时预测
2. **启动预测**：点击「启动预测流程」按钮，系统开始训练模型并生成预测
3. **查看结果**：预测完成后，查看预测图表和统计信息
4. **生成简报**：选择 AI 大模型，点击「生成简报」生成分析报告
5. **导出报告**：支持导出 Markdown、HTML、TXT 格式的报告

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/forecast/` | POST | 启动预测流程 |
| `/api/forecast/` | GET | 获取预测结果 |
| `/api/generate_report/` | POST | 生成 AI 分析简报 |
| `/api/export_report/` | POST | 导出报告 |
| `/api/models/` | GET | 获取已保存模型列表 |

### 示例请求

```bash
# 启动预测（24小时）
curl -X POST http://127.0.0.1:8000/api/forecast/ \
  -H "Content-Type: application/json" \
  -d '{"forecast_hours": 24}'

# 生成简报（使用百度千帆模型）
curl -X POST http://127.0.0.1:8000/api/generate_report/ \
  -H "Content-Type: application/json" \
  -d '{"model": "qianfan"}'
```

## 📁 项目结构

```
load_forecasting-master/
├── _archived_unused_files/       # 归档的无关文件
├── models/                       # 模型实验代码
│   ├── LSTM.ipynb               # LSTM 模型实验
│   ├── GRU.ipynb                # GRU 模型实验
│   ├── ARIMA.ipynb              # ARIMA 模型实验
│   └── utils.py                 # 工具函数
├── saved_models/                 # 保存的模型权重
├── screenshots/                  # Web 界面截图
├── server/                       # Django 服务器
│   ├── swag/                    # 核心业务逻辑
│   │   ├── views.py             # API 视图
│   │   ├── models.py            # 数据库模型
│   │   └── urls.py              # 路由配置
│   ├── tamplates/               # HTML 模板
│   │   ├── Home_page.html       # 欢迎页
│   │   ├── forecast_dashboard.html  # 预测仪表板
│   │   └── form.html            # 登录/注册页
│   ├── static/                  # 静态资源
│   ├── users/                   # 用户管理模块
│   ├── website/                 # Django 配置
│   └── manage.py                # Django 管理命令
├── generate_report.py            # AI 报告生成核心
├── myModel.lstm.py              # LSTM 模型核心
├── test_features.py             # 功能测试脚本
├── run_full_test.py             # 完整测试流程
├── start_server.py              # Python 启动器
├── start_server.bat             # Windows 启动脚本
├── requirements.txt             # 依赖清单
├── .env                         # 环境变量配置（已加入 .gitignore）
├── .gitignore                   # Git 忽略规则
├── 技术路线书.md                 # 技术路线文档
└── 系统使用指南.md               # 使用说明文档
```

## 🤖 大模型配置

### 支持的模型

| 模型 | Key | 说明 |
|------|-----|------|
| 百度千帆 | `qianfan` | ERNIE-4.0 大模型，中文理解能力强 |
| 阿里通义千问 | `dashscope` | Qwen-Plus 模型，推理速度快 |
| 字节豆包 | `doubao` | Doubao-3 模型，响应迅速 |
| 科大讯飞星火 | `xunfei` | Spark-4.0 模型，专业领域能力强 |

### 配置示例

```bash
# .env 文件

# 百度千帆
QIANFAN_API_KEY=your_qianfan_api_key
QIANFAN_MODEL=ernie-4.0-8k-latest

# 阿里通义千问
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_MODEL=qwen-plus

# 字节豆包
DOUBAN_API_KEY=your_douban_api_key
DOUBAN_MODEL=Doubao-3-4k

# 科大讯飞星火
XUNFEI_APP_ID=your_xunfei_app_id
XUNFEI_API_KEY=your_xunfei_api_key
XUNFEI_API_SECRET=your_xunfei_api_secret
XUNFEI_MODEL=spark-4.0
```

## 📊 数据说明

### 数据源

项目使用 `powerdemand_5min_2021_to_2024_with weather.csv` 数据集，包含：

- **负荷数据**：5分钟粒度的电力负荷值（MW）
- **气象数据**：温度、湿度、风速等
- **时间特征**：日期、小时、分钟、星期几等

### 特征工程

系统自动进行以下特征处理：

- 时间编码（小时、分钟、星期、节假日）
- 气象特征归一化
- 历史负荷滞后特征
- 滚动统计特征

## 🧪 测试

```bash
# 功能测试
python test_features.py

# 完整流程测试
python run_full_test.py
```

## 📝 许可证

MIT License

## 📧 联系

如有问题或建议，请联系项目维护者。