import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

output_lines = []
output_lines.append(f"API Key: {api_key[:20]}..." if api_key else "API Key: None")
output_lines.append(f"Base URL: {base_url}")
output_lines.append(f"Model: {model}")
output_lines.append("")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"}
        ],
        temperature=0.3
    )
    content = response.choices[0].message.content
    output_lines.append("SUCCESS")
    output_lines.append(f"Response: {content[:200]}")
except Exception as e:
    output_lines.append(f"ERROR_TYPE: {type(e).__name__}")
    output_lines.append(f"ERROR_MSG: {str(e)[:200]}")
    if hasattr(e, 'response'):
        output_lines.append(f"STATUS_CODE: {e.response.status_code}")
        try:
            text = e.response.text[:500]
            output_lines.append(f"RESPONSE: {text}")
        except:
            output_lines.append("RESPONSE: Cannot read")

result = "\n".join(output_lines)

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write(result)