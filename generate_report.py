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
# 1. 模拟前置 PyTorch 模型的输出结果
# ==========================================
def get_model_outputs():
    """
    在实际业务中，这部分数据来自您之前的 PyTorch LSTM 模型和异常检测引擎。
    这里我们构造一个标准的字典和DataFrame来模拟输出。
    """
    # 预测统计指标 (未来24小时)
    forecast_stats = {
        "预测日期": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "预测时间粒度": "5分钟/点 (共288个点)",
        "预测最大负荷(MW)": 5240.5,
        "预计峰值出现时间": "18:45 - 19:15",
        "预测最小负荷(MW)": 1620.3,
        "预计谷值出现时间": "03:30 - 04:00",
        "平均负荷(MW)": 3450.2,
        "峰谷差(MW)": 3620.2,
        "95%置信区间最大宽度(MW)": 215.0,  # 反映预测不确定性
        "气象关联特征": "明日有强冷空气南下，预计降温8-10度，伴有5级阵风"
    }

    # 异常识别结果 (来自任务三的规则引擎)
    anomalies_data = [
        {"时间": "07:30", "异常类型": "负荷突增", "实际/预测负荷": 3150.0, "偏离幅度": "+450 MW",
         "风险等级": "高(红色)"},
        {"时间": "14:15", "异常类型": "持续偏高(趋势偏离)", "实际/预测负荷": 4850.0, "偏离幅度": "+12.5%",
         "风险等级": "中(橙色)"},
        {"时间": "21:00", "异常类型": "负荷突降", "实际/预测负荷": 2800.0, "偏离幅度": "-380 MW",
         "风险等级": "高(红色)"}
    ]
    anomalies_df = pd.DataFrame(anomalies_data)

    return forecast_stats, anomalies_df


# ==========================================
# 2. 构建专业 Prompt (提示词工程)
# ==========================================
def build_analyst_prompt(stats, anomalies_df):
    """
    构建赋予大模型“资深电网调度分析师”角色的系统级Prompt
    """
    # 将异常DataFrame转为文本格式，方便大模型阅读
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
    """
    检查网络是否可以连接到百度千帆服务器
    
    参数:
        host: 目标主机名
        port: 目标端口
        timeout: 超时时间（秒）
    
    返回:
        (bool, str): (是否可达, 详细信息)
    """
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
# 3. 调用百度千帆 (文心一言) 智能体
# ==========================================
def call_qianfan_agent(prompt, debug_mode=None, max_retries=None, timeout=None):
    debug_mode = debug_mode if debug_mode is not None else (os.getenv("LOG_LEVEL") == "DEBUG")
    max_retries = max_retries if max_retries is not None else int(os.getenv("MAX_RETRIES", "3"))
    timeout = timeout if timeout is not None else int(os.getenv("API_TIMEOUT", "30"))
    """
    通过 OpenAI 兼容接口调用百度千帆大模型 (ERNIE-4.0 或 ERNIE-3.5)
    
    参数:
        prompt: 提示词内容
        debug_mode: 是否开启调试模式，开启后会输出详细的调试信息
        max_retries: 最大重试次数
        timeout: 请求超时时间（秒）
    
    返回:
        生成的报告内容，或错误信息
    """
    # ⚠️ 配置说明：
    # 1. 登录百度千帆大模型平台 (https://qianfan.cloud.baidu.com/)
    # 2. 获取您的 API Key (格式通常为 bce-v3/xxx 或直接的字符串)
    # 3. 将下方 "your_qianfan_api_key" 替换为您的真实 Key，或设置环境变量 QIANFAN_API_KEY
    
    # 从环境变量读取配置
    QIANFAN_BASE_URL = os.getenv("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2")
    QIANFAN_MODEL = os.getenv("QIANFAN_MODEL", "ernie-4.0-8k-latest")
    
    # 获取 API Key（优先从环境变量读取）
    api_key = os.getenv("QIANFAN_API_KEY")
    
    # 调试信息：打印当前配置状态
    if debug_mode:
        print(f"\n[DEBUG] 开始调试千帆大模型调用...")
        print(f"[DEBUG] API Key 来源: {'环境变量' if api_key else '默认值'}")
        print(f"[DEBUG] API Key 长度: {len(api_key) if api_key else 0} 字符")
        print(f"[DEBUG] Base URL: {QIANFAN_BASE_URL}")
        print(f"[DEBUG] Model: {QIANFAN_MODEL}")
        print(f"[DEBUG] Prompt 长度: {len(prompt)} 字符")
        print(f"[DEBUG] 最大重试次数: {max_retries}")
        print(f"[DEBUG] 请求超时时间: {timeout}秒")
    
    # 检查 API Key 是否配置
    if not api_key or api_key.strip() == "" or api_key == "your_qianfan_api_key_here":
        if not api_key or api_key.strip() == "":
            print("[WARNING] 未配置百度千帆 API Key，将生成模拟简报")
            print("[INFO] 请在 .env 文件中设置 QIANFAN_API_KEY 以使用真实大模型")
        else:
            print("[WARNING] 检测到占位符 API Key，将生成模拟简报")
            print("[INFO] 请在 .env 文件中替换为真实的 API Key")
        return generate_mock_report()

    # 网络预检查
    print("\n[INFO] 正在进行网络连接检查...")
    network_ok, network_msg = check_network_connectivity("qianfan.baidubce.com", 443, 5)
    print(f"[INFO] {network_msg}")
    
    if not network_ok:
        error_msg = f"❌ 网络连接失败: {network_msg}"
        error_detail = "\n\n详细原因分析:\n1. 当前网络环境无法访问百度千帆服务器\n2. 防火墙或代理设置可能阻止了请求\n3. 请检查网络连接和防火墙设置\n4. 在浏览器中访问 https://qianfan.cloud.baidu.com/ 确认网络可达"
        print(error_msg)
        print(error_detail)
        return error_msg + error_detail + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()

    # 初始化 OpenAI 客户端，指向百度千帆的 Base URL
    client = OpenAI(
        api_key=api_key,
        base_url=QIANFAN_BASE_URL,
        timeout=timeout
    )
    
    if debug_mode:
        print(f"[DEBUG] OpenAI 客户端初始化成功")
        print(f"[DEBUG] 客户端配置: api_key已设置, base_url={client.base_url}, timeout={timeout}秒")
    
    # 重试循环
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"\n[INFO] 正在调用千帆大模型智能体 (第 {attempt + 1}/{max_retries} 次尝试)...")
            
            # 调用大模型
            print(f"[INFO] 请求模型: {QIANFAN_MODEL}")
            print(f"[INFO] 请求参数: temperature=0.3, top_p=0.8")
            
            response = client.chat.completions.create(
                model=QIANFAN_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严谨的电网调度分析师，严格遵循Markdown格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.8
            )
            
            if debug_mode:
                print(f"[DEBUG] API 响应成功")
                print(f"[DEBUG] 响应对象类型: {type(response)}")
                print(f"[DEBUG] 响应 ID: {response.id if hasattr(response, 'id') else 'N/A'}")
            
            # 提取响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    print(f"[INFO] 成功获取到报告内容，长度: {len(message.content)} 字符")
                    return message.content
                else:
                    error_msg = f"❌ API 响应内容为空: message.content = {message.content}"
                    print(error_msg)
                    last_error = error_msg
            else:
                error_msg = f"❌ API 响应中没有 choices 字段或 choices 为空"
                print(error_msg)
                last_error = error_msg
        
        except AuthenticationError as e:
            error_msg = f"❌ 认证错误 (AuthenticationError): {str(e)}"
            error_detail = "\n详细原因分析:\n1. API Key 可能不正确或已过期\n2. API Key 格式可能有误（应为 bce-v3/xxx 格式）\n3. 账号可能未开通千帆 API 服务\n4. 请检查百度千帆控制台中的 API Key 配置"
            print(error_msg)
            print(error_detail)
            return error_msg + error_detail + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()
        
        except APIConnectionError as e:
            error_msg = f"❌ 连接错误 (APIConnectionError): {str(e)}"
            print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)
        
        except RateLimitError as e:
            error_msg = f"❌ 限流错误 (RateLimitError): {str(e)}"
            error_detail = "\n详细原因分析:\n1. API 请求频率超过了限制\n2. 账号配额已用完\n3. 请稍后重试或联系百度千帆客服增加配额"
            print(error_msg)
            print(error_detail)
            return error_msg + error_detail + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()
        
        except APIError as e:
            error_msg = f"❌ API 错误 (APIError): {str(e)}"
            
            if 'account_overdue' in str(e):
                error_detail = "\n详细原因分析:\n⚠️ 【重要】百度千帆账户欠费！\n1. 当前使用的API Key对应的百度千帆账户已欠费\n2. 请登录百度千帆控制台 (https://qianfan.cloud.baidu.com/) 检查账户余额\n3. 充值后即可恢复服务\n4. 如需更换API Key，请设置环境变量 QIANFAN_API_KEY"
                print(error_msg)
                print(error_detail)
                return error_msg + error_detail + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()
            else:
                print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(wait_time)
        
        except Exception as e:
            error_msg = f"❌ 未知错误: {str(e)}"
            print(f"[WARNING] 第 {attempt + 1} 次尝试失败: {error_msg}")
            if debug_mode:
                print("[DEBUG] 完整错误堆栈:")
                traceback.print_exc()
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] {wait_time}秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)
    
    # 所有重试都失败了
    error_summary = f"❌ 所有 {max_retries} 次请求均失败，无法连接到百度千帆大模型服务。"
    if last_error:
        error_summary += f"\n最后一次错误: {str(last_error)}"
    
    error_detail = "\n\n详细原因分析:\n1. 网络连接问题，无法连接到百度千帆服务器\n2. 防火墙或代理设置可能阻止了请求\n3. 请检查网络连接和防火墙设置\n4. 尝试访问 https://qianfan.cloud.baidu.com/ 确认网络可达\n5. 请确认百度千帆账号余额充足且API Key有效"
    
    print(error_summary)
    print(error_detail)
    return error_summary + error_detail + "\n\n以下为基于规则生成的【模拟简报】：\n" + generate_mock_report()


def generate_mock_report():
    """当没有API Key时，提供一个高质量的模板示例"""
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
# 4. 主程序执行
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("⚡ 区域电网负荷预测与运行风险智能分析系统 ⚡")
    print("=" * 50)

    # 1. 获取前置模型结果
    stats, anomalies = get_model_outputs()

    # 2. 构建 Prompt
    prompt = build_analyst_prompt(stats, anomalies)

    # 3. 调用大模型生成简报
    report = call_qianfan_agent(prompt)

    # 4. 输出结果
    print("\n" + "=" * 50)
    print("📝 【生成简报如下】")
    print("=" * 50)
    print(report)

    # 可选：将简报保存为 Markdown 文件
    with open("forecast_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n✅ 简报已保存至 forecast_report.md")