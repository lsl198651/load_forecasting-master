import json
import os
import sys
import time
import traceback
import pandas as pd
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError, RateLimitError
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] 成功加载 .env 配置文件")
except ImportError:
    print("[INFO] 未安装 python-dotenv，将从环境变量读取配置")

# ==========================================
# 模型配置管理器 - 支持多模型切换
# ==========================================
class ModelManager:
    MODELS = {
        "qianfan": {
            "name": "百度千帆",
            "description": "ERNIE-4.0 大模型，中文理解能力强",
            "base_url": os.getenv("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2"),
            "model": os.getenv("QIANFAN_MODEL", "ernie-4.0-8k-latest"),
            "api_key_env": "QIANFAN_API_KEY",
            "test_host": "qianfan.baidubce.com",
            "free": True,
            "provider": "baidu"
        },
        "dashscope": {
            "name": "阿里百炼",
            "description": "Qwen-Plus 模型，推理速度快",
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
            "api_key_env": "DASHSCOPE_API_KEY",
            "test_host": "dashscope.aliyuncs.com",
            "free": True,
            "provider": "alibaba"
        },
        "doubao": {
            "name": "字节豆包",
            "description": "Doubao-3 模型，响应迅速",
            "base_url": os.getenv("DOUBAN_BASE_URL", "https://api.doubao.com/v1"),
            "model": os.getenv("DOUBAN_MODEL", "Doubao-3-4k"),
            "api_key_env": "DOUBAN_API_KEY",
            "test_host": "api.doubao.com",
            "free": True,
            "provider": "bytedance"
        },
        "xunfei": {
            "name": "科大讯飞星火",
            "description": "Spark-4.0 模型，专业领域能力强",
            "base_url": "https://spark-api.xf-yun.com/v4/chat/completions",
            "model": os.getenv("XUNFEI_MODEL", "spark-4.0"),
            "api_key_env": "XUNFEI_API_KEY",
            "app_id_env": "XUNFEI_APP_ID",
            "api_secret_env": "XUNFEI_API_SECRET",
            "test_host": "spark-api.xf-yun.com",
            "free": True,
            "provider": "xunfei",
            "special": True
        }
    }

    @classmethod
    def get_available_models(cls):
        """获取所有可用模型列表"""
        return {key: cls.MODELS[key] for key in cls.MODELS}

    @classmethod
    def get_model_config(cls, model_key):
        """获取指定模型的配置"""
        return cls.MODELS.get(model_key)

    @classmethod
    def get_default_model(cls):
        """获取默认模型"""
        default_key = os.getenv("DEFAULT_MODEL", "qianfan")
        return default_key, cls.MODELS.get(default_key, cls.MODELS["qianfan"])


# ==========================================
# 1. 模拟前置 PyTorch 模型的输出结果
# ==========================================
def get_model_outputs():
    forecast_stats = {
        "预测日期": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "预测时间粒度": "5分钟/点 (共288个点)",
        "预测最大负荷(MW)": 5240.5,
        "预计峰值出现时间": "18:45 - 19:15",
        "预测最小负荷(MW)": 1620.3,
        "预计谷值出现时间": "03:30 - 04:00",
        "平均负荷(MW)": 3450.2,
        "峰谷差(MW)": 3620.2,
        "95%置信区间最大宽度(MW)": 215.0,
        "气象关联特征": "明日有强冷空气南下，预计降温8-10度，伴有5级阵风"
    }

    anomalies_data = [
        {"时间": "07:30", "异常类型": "负荷突增", "实际/预测负荷": 3150.0, "偏离幅度": "+450 MW", "风险等级": "高(红色)"},
        {"时间": "14:15", "异常类型": "持续偏高(趋势偏离)", "实际/预测负荷": 4850.0, "偏离幅度": "+12.5%", "风险等级": "中(橙色)"},
        {"时间": "21:00", "异常类型": "负荷突降", "实际/预测负荷": 2800.0, "偏离幅度": "-380 MW", "风险等级": "高(红色)"}
    ]
    anomalies_df = pd.DataFrame(anomalies_data)

    return forecast_stats, anomalies_df


# ==========================================
# 2. 构建专业 Prompt
# ==========================================
def build_analyst_prompt(stats, anomalies_df):
    anomalies_text = anomalies_df.to_string(index=False) if not anomalies_df.empty else "无异常状态，预测曲线平滑贴合历史规律。"

    prompt = f"""
# 角色设定
你是一位拥有20年经验的省级电网调度运行资深分析师。你精通电力系统运行规程、负荷特性分析、气象对电网的影响以及机组调度策略。

# 任务目标
请根据以下提供的【未来24小时负荷预测核心指标】、【气象特征】与【AI异常状态识别结果】，撰写一份面向电网调度台和运行方式处的《区域电网负荷预测与运行风险简报》。

# 输入数据
【预测核心指标】：
{json.dumps(stats, ensure_ascii=False, indent=2)}

【AI异常状态识别结果】：
{anomalies_text}

# 输出要求与格式规范
请使用 Markdown 格式输出，语言必须专业、严谨、精炼，符合电网调度术语规范。简报必须严格包含以下三个模块：

## 📊 一、 关键结论
1. 概述明日整体负荷水平及峰谷差情况。
2. 结合气象特征（如降温/升温），分析负荷趋势的合理性及AI预测置信度（参考置信区间宽度）。
3. 明确指出早晚高峰的保供压力时段。

## ⚠️ 二、 风险提示
1. 针对【AI异常状态识别结果】中的“突增/突降”或“趋势偏离”事件，分析其对电网可能造成的物理冲击（如：机组爬坡速率跟不上、局部断面潮流越限、调峰困难等）。
2. 指出高风险（红色）时段需要重点盯防的调度断面或变电站。

## 💡 三、 运行建议
基于上述结论和风险，向当值调度员提出具体的、可操作的运行建议（包括但不限于）：
1. **机组组合与备用**：抽水蓄能、燃气机组的启停建议，系统旋转备用容量安排。
2. **断面控制**：针对突增/突降时段的潮流控制预案。
3. **需求侧响应**：是否需要提前联系大用户或启动有序用电/需求侧响应预案。
4. **新能源消纳**：结合负荷谷值与突降情况，对风光新能源消纳的预警。
"""
    return prompt


# ==========================================
# 3. 网络预检查函数
# ==========================================
def check_network_connectivity(host="qianfan.baidubce.com", port=443, timeout=5):
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return True, f"✅ 网络连接正常，{host}:{port} 可达"
        else:
            return False, f"❌ 网络连接失败，{host}:{port} 不可达 (错误码: {result})"
    except socket.gaierror:
        return False, f"❌ DNS解析失败，无法解析主机名 {host}"
    except socket.timeout:
        return False, f"❌ 连接超时，{host}:{port} 响应时间超过 {timeout} 秒"
    except Exception as e:
        return False, f"❌ 网络检查失败: {str(e)}"


# ==========================================
# 4. 科大讯飞星火专用调用函数
# ==========================================
def call_xunfei_spark(prompt, app_id, api_key, api_secret, model="spark-4.0", timeout=30, debug_mode=True):
    try:
        import websocket
        import base64
        import hashlib
        import hmac
        import json
        from datetime import datetime
        from urllib.parse import urlencode

        if not app_id or not api_key or not api_secret:
            print("[WARNING] 科大讯飞 API 配置不完整")
            return generate_mock_report()

        url = "wss://spark-api.xf-yun.com/v4/chat/completions"
        host = "spark-api.xf-yun.com"

        date = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        signature_origin = f"host: {host}\ndate: {date}\nGET /v4/chat/completions HTTP/1.1"
        signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        params = {
            "authorization": authorization,
            "date": date,
            "host": host
        }

        ws_url = f"{url}?{urlencode(params)}"

        ws = websocket.create_connection(ws_url, timeout=timeout)

        data = {
            "header": {"app_id": app_id},
            "parameter": {
                "chat": {
                    "domain": model,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "system", "content": "你是一个严谨的电网调度分析师，严格遵循Markdown格式输出。"},
                        {"role": "user", "content": prompt}
                    ]
                }
            }
        }

        ws.send(json.dumps(data))

        result = ""
        while True:
            response = ws.recv()
            response_data = json.loads(response)
            if response_data.get("header", {}).get("code") != 0:
                error_msg = f"❌ 科大讯飞 API 错误: {response_data.get('header', {}).get('message', '未知错误')}"
                print(error_msg)
                ws.close()
                return error_msg + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()

            choices = response_data.get("payload", {}).get("choices", {}).get("text", [])
            for choice in choices:
                if choice.get("content"):
                    result += choice.get("content", "")

            if response_data.get("payload", {}).get("choices", {}).get("status") == 2:
                break

        ws.close()
        if debug_mode:
            print(f"[INFO] 成功获取到报告内容，长度: {len(result)} 字符")
        return result

    except ImportError:
        print("[WARNING] 未安装 websocket-client，无法调用科大讯飞星火模型")
        return generate_mock_report()
    except Exception as e:
        print(f"❌ 科大讯飞调用错误: {str(e)}")
        return generate_mock_report()


# ==========================================
# 5. 统一调用接口 - 支持多模型
# ==========================================
def call_ai_model(prompt, model_key=None, debug_mode=None, max_retries=None, timeout=None):
    debug_mode = debug_mode if debug_mode is not None else (os.getenv("LOG_LEVEL") == "DEBUG")
    max_retries = max_retries if max_retries is not None else int(os.getenv("MAX_RETRIES", "3"))
    timeout = timeout if timeout is not None else int(os.getenv("API_TIMEOUT", "30"))

    if model_key is None:
        model_key = os.getenv("DEFAULT_MODEL", "qianfan")

    model_config = ModelManager.get_model_config(model_key)
    if not model_config:
        print(f"[WARNING] 未找到模型配置: {model_key}，使用默认模型")
        model_key = "qianfan"
        model_config = ModelManager.MODELS["qianfan"]

    if debug_mode:
        print(f"\n[DEBUG] 开始调用大模型: {model_config['name']}")
        print(f"[DEBUG] 模型描述: {model_config['description']}")
        print(f"[DEBUG] Base URL: {model_config['base_url']}")
        print(f"[DEBUG] Model: {model_config['model']}")

    api_key = os.getenv(model_config["api_key_env"])

    if not api_key or api_key.strip() == "" or "your_" in api_key:
        print(f"[WARNING] 未配置 {model_config['name']} API Key，将生成模拟简报")
        print(f"[INFO] 请在 .env 文件中设置 {model_config['api_key_env']}")
        return generate_mock_report()

    if model_config.get("special") and model_key == "xunfei":
        app_id = os.getenv(model_config.get("app_id_env"))
        api_secret = os.getenv(model_config.get("api_secret_env"))
        return call_xunfei_spark(prompt, app_id, api_key, api_secret, model_config["model"], timeout, debug_mode)

    print(f"\n[INFO] 正在进行网络连接检查...")
    network_ok, network_msg = check_network_connectivity(model_config["test_host"], 443, 5)
    print(f"[INFO] {network_msg}")

    if not network_ok:
        error_msg = f"❌ 网络连接失败: {network_msg}"
        print(error_msg)
        return error_msg + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()

    client = OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
        timeout=timeout
    )

    if debug_mode:
        print(f"[DEBUG] OpenAI 客户端初始化成功")

    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"\n[INFO] 正在调用 {model_config['name']} 大模型 (第 {attempt + 1}/{max_retries} 次尝试)...")
            print(f"[INFO] 请求模型: {model_config['model']}")

            response = client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个严谨的电网调度分析师，严格遵循Markdown格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.8
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    print(f"[INFO] 成功获取到报告内容，长度: {len(message.content)} 字符")
                    return message.content
                else:
                    error_msg = f"❌ API 响应内容为空"
                    print(error_msg)
                    last_error = error_msg
            else:
                error_msg = f"❌ API 响应中没有 choices 字段"
                print(error_msg)
                last_error = error_msg

        except AuthenticationError as e:
            error_msg = f"❌ 认证错误: {str(e)}"
            print(error_msg)
            return error_msg + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()

        except APIConnectionError as e:
            error_msg = f"❌ 连接错误: {str(e)}"
            print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)

        except RateLimitError as e:
            error_msg = f"❌ 限流错误: {str(e)}"
            print(error_msg)
            return error_msg + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()

        except APIError as e:
            error_msg = f"❌ API 错误: {str(e)}"
            print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
            if debug_mode:
                import httpx
                actual_url = f"{model_config['base_url']}/chat/completions"
                print(f"[DEBUG] 实际请求 URL: {actual_url}")
                print(f"[DEBUG] 请求模型: {model_config['model']}")
                print(f"[DEBUG] 完整错误信息: {e}")
                if hasattr(e, 'response'):
                    print(f"[DEBUG] 响应状态码: {e.response.status_code}")
                    try:
                        print(f"[DEBUG] 响应内容: {e.response.text[:2000]}")
                    except:
                        pass
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)

        except Exception as e:
            error_msg = f"❌ 未知错误: {str(e)}"
            print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
            if debug_mode:
                traceback.print_exc()
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)

    error_summary = f"❌ 所有 {max_retries} 次请求均失败"
    if last_error:
        error_summary += f"\n最后一次错误: {str(last_error)}"
    print(error_summary)
    return error_summary + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()


# ==========================================
# 6. 兼容旧接口
# ==========================================
def call_qianfan_agent(prompt, debug_mode=None, max_retries=None, timeout=None):
    return call_ai_model(prompt, "qianfan", debug_mode, max_retries, timeout)


# ==========================================
# 7. 模拟报告生成
# ==========================================
def generate_mock_report():
    return """## 📊 一、 关键结论
1. **整体负荷偏高**：明日受强冷空气南下影响，取暖负荷激增，预计最大负荷达 5240.5 MW，峰谷差高达 3620.2 MW，系统调峰压力极大。
2. **峰值时段集中**：晚高峰 18:45-19:15 为全网保供最严峻时段，与光伏出力骤降期重叠（“鸭子曲线”颈部），需高度警惕。
3. **预测置信度**：AI模型95%置信区间最大宽度达 215 MW，表明在极端天气下负荷波动性增强，实际运行需留有更大裕度。

## ⚠️ 二、 风险提示
1. **早高峰爬坡风险 (07:30)**：识别到 +450 MW 的负荷突增（红色风险）。此时段正值早高峰通勤与工业复产叠加，若系统备用不足，可能导致 ACE（区域控制误差）超标，甚至引发低频减载。
2. **晚高峰断面越限 (14:15 持续偏高预警)**：午后负荷持续偏高 12.5%，可能导致局部重载断面（如某某输电通道）潮流逼近稳定极限。
3. **夜间调峰困难 (21:00)**：识别到 -380 MW 的负荷突降（红色风险）。夜间风电若处于大发期，叠加负荷骤降，极易引发新能源弃风弃光及系统频率偏高。

## 💡 三、 运行建议
1. **机组组合安排**：建议今夜提前启动 2-3 台燃气机组或安排抽水蓄能机组处于旋转备用状态，确保明日 07:00 前系统具备至少 600 MW 的向上爬坡能力。
2. **断面潮流控制**：方式处需重新校核 14:00 时段的重载断面限额，必要时调整电网运行方式，解环部分电磁环网。
3. **需求侧响应**：建议今日 16:00 前向高耗能企业发布晚高峰错峰生产预警，准备调用 200 MW 需求侧响应资源。
4. **新能源消纳**：针对 21:00 的负荷突降风险，提前与省间现货市场沟通，扩大新能源外送通道，或安排部分火电机组深度调峰（降至 40% 出力）。"""


# ==========================================
# 8. 主程序执行
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("⚡ 区域电网负荷预测与运行风险智能分析系统 ⚡")
    print("=" * 50)

    print("\n[可用模型列表]")
    for key, config in ModelManager.get_available_models().items():
        print(f"  - {key}: {config['name']} ({config['description']})")

    stats, anomalies = get_model_outputs()
    prompt = build_analyst_prompt(stats, anomalies)

    print("\n[测试调用默认模型]")
    report = call_ai_model(prompt)

    print("\n" + "=" * 50)
    print("📝 【生成简报如下】")
    print("=" * 50)
    print(report)

    with open("forecast_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n✅ 简报已保存至 forecast_report.md")