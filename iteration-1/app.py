from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import datetime
import threading
import time
from openai import OpenAI  # 使用OpenAI SDK

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 这里使用OpenRouter API密钥
API_KEY = ""

# 初始化OpenAI客户端，连接到OpenRouter API
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

@app.route('/search', methods=['POST'])
def search_files():
    data = request.get_json(force=True) or {}
    user_instruction = data.get('instruction')
    
    if not user_instruction:
        return jsonify({"error": "请输入搜索指令"}), 400
    
    # 调用 LLM 处理指令
    search_criteria = process_instruction_with_deepseek(user_instruction)
    
    # 如果返回的 search_criteria 中包含错误信息，则直接返回错误
    if search_criteria.get("error"):
        return jsonify(search_criteria)
    
    directory_path = data.get('directory', '/path/to/default/directory')
    results = search_local_files(directory_path, search_criteria)
    
    return jsonify({"results": results})

def process_instruction_with_deepseek(instruction):
    """
    使用 OpenRouter API 调用 LLM 解析用户指令，返回搜索条件。
    使用线程和超时机制防止长时间等待。
    如果出现错误，则返回 {"error": "错误信息"}。
    """
    response_received = {"status": False, "data": None, "error": None}

    def call_api():
        try:
            print(f"[DEBUG] 正在调用API，指令：{instruction}")
            response = client.chat.completions.create(
                model="deepseek/deepseek-r1-distill-qwen-14b:free",
                messages=[
                    {"role": "system", "content": "分析用户想搜索什么文件类型。只返回文件扩展名列表，使用逗号分隔，例如：py,doc,pdf"},
                    {"role": "user", "content": instruction}
                ],
                stream=False
            )
            print(f"[DEBUG] API调用成功，完整返回：{response}")
            response_received["data"] = response
        except Exception as e:
            print(f"[ERROR] API调用异常: {e}")
            response_received["error"] = str(e)
        finally:
            response_received["status"] = True

    api_thread = threading.Thread(target=call_api)
    api_thread.start()

    timeout = 10  # 最多等待10秒
    start_time = time.time()
    while not response_received["status"] and time.time() - start_time < timeout:
        time.sleep(0.5)

    if not response_received["status"]:
        error_msg = "API请求超时，请稍后重试"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

    if response_received["error"]:
        error_msg = f"API调用错误: {response_received['error']}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

    if not response_received["data"]:
        error_msg = "未收到API返回数据"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

    content = response_received["data"].choices[0].message.content
    print(f"[DEBUG] API返回内容原始数据: {content}")

    # 将返回内容转换为小写，并拆分成列表（按逗号分隔）
    content_lower = content.lower().strip()
    raw_types = [ft.strip() for ft in content_lower.split(',')]
    print(f"[DEBUG] 拆分后的文件类型列表: {raw_types}")

    type_mapping = {
        "py": "py",
        "python": "py",
        "doc": "doc",
        "docx": "doc",
        "word": "doc",
        "pdf": "pdf",
        "txt": "txt",
        "文本": "txt",
        "html": "html",
        "js": "js",
        "javascript": "js"
    }
    
    file_types = []
    for ft in raw_types:
        if ft in type_mapping:
            mapped = type_mapping[ft]
            if mapped not in file_types:
                file_types.append(mapped)
    
    print(f"[DEBUG] 最终解析得到的文件类型: {file_types}")

    if not file_types:
        error_msg = "无法识别所需的文件类型"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}
        
    return {"file_types": file_types, "content_search": False}

def search_local_files(directory, criteria):
    """
    在指定目录下搜索符合 criteria 中 file_types 的文件，
    返回文件名、路径、大小、修改时间以及预览信息。
    """
    results = []
    # 如果 criteria 没有 file_types 字段，这里提供默认值
    file_types = criteria.get('file_types', ["py"])
    
    if not os.path.exists(directory):
        print(f"[WARNING] 指定的目录不存在: {directory}")
        return results

    for root, _, files in os.walk(directory):
        for file in files:
            # 判断文件扩展名是否匹配
            if any(file.lower().endswith(f".{ft.lower()}") for ft in file_types):
                file_path = os.path.join(root, file)
                file_info = {
                    "file_name": file,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                }
                try:
                    mod_time = os.path.getmtime(file_path)
                    file_info["modified"] = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"[ERROR] 获取文件修改时间出错: {file_path}, error: {e}")
                # 对于 Python 文件额外获取预览信息
                if file.lower().endswith(".py"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            preview_lines = content.split('\n')[:5]
                            file_info["preview"] = "\n".join(preview_lines) + "\n..."
                    except Exception as e:
                        print(f"[WARNING] 读取文件 {file_path} 时出错: {e}")
                results.append(file_info)
    return results

if __name__ == '__main__':
    app.run(debug=True, port=5000)
