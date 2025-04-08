import json
import subprocess
import sys
import time
import os
from openai import OpenAI

# 使用提供的OpenRouter API密钥
API_KEY = ""

# 初始化OpenAI客户端，连接到OpenRouter API
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# 彩色输出函数
def print_info(message):
    print(f"\033[94m[INFO]\033[0m {message}")

def print_success(message):
    print(f"\033[92m[SUCCESS]\033[0m {message}")

def print_error(message):
    print(f"\033[91m[ERROR]\033[0m {message}")

def print_warning(message):
    print(f"\033[93m[WARNING]\033[0m {message}")

# 函数定义
def train_yolo(model_type='yolov8n', epochs=1, data='coco128.yaml'):
    """训练YOLOv8模型"""
    try:
        print_info(f"准备训练 {model_type} 模型，使用 {data} 数据集，训练 {epochs} 轮...")
        print_info("正在启动训练进程...")
        
        # 设置环境变量解决编码问题
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # 使用修改后的环境变量启动进程
        process = subprocess.Popen(
            ['python', 'train.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace',  # 添加此行处理编码错误
            env=env
        )
        
        print_success("训练进程已启动，实时输出如下:")
        print("-" * 50)
        
        # 实时读取输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()
        
        # 检查错误
        stderr = process.stderr.read()
        return_code = process.poll()
        
        if return_code != 0:
            print_error(f"训练出错 (返回码: {return_code})")
            if stderr:
                print_error(f"错误信息: {stderr}")
            return {"status": "error", "message": f"训练出错: {stderr}"}
        else:
            print_success("训练已成功启动！模型训练进程正在进行中...")
            return {"status": "success", "message": "YOLO模型训练已成功启动"}
    except Exception as e:
        print_error(f"执行异常: {str(e)}")
        return {"status": "error", "message": f"执行异常: {str(e)}"}

# 函数映射
FUNCTION_MAP = {
    "train_yolo": train_yolo
}

# 函数声明，用于LLM的函数调用
FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "train_yolo",
            "description": "训练YOLOv8模型",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "YOLO模型类型，默认为yolov8n"
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "训练轮数，默认为20"
                    },
                    "data": {
                        "type": "string",
                        "description": "数据集配置文件，默认为coco128.yaml"
                    }
                },
                "required": []
            }
        }
    }
]

def process_instruction(instruction):
    """使用LLM处理用户指令并转换为函数调用"""
    try:
        print_info(f"正在分析指令: '{instruction}'")
        print_info("正在连接到AI服务...")
        
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo:free",
            messages=[
                {"role": "system", "content": "你是一个智能助手，帮助用户将自然语言指令转换为函数调用。"},
                {"role": "user", "content": instruction}
            ],
            tools=FUNCTIONS
        )
        
        print_success("AI服务返回成功")
        print_info(f"原始响应: {response}")
        
        # 检查是否有函数调用
        if response.choices and response.choices[0].message and hasattr(response.choices[0].message, 'tool_calls'):
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print_info(f"指令已解析为函数: {function_name}")
            print_info(f"函数参数: {arguments}")
            
            # 查找并调用对应的函数
            if function_name in FUNCTION_MAP:
                function = FUNCTION_MAP[function_name]
                print_info(f"正在执行函数: {function_name}")
                return function(**arguments)
            else:
                print_error(f"未知函数: {function_name}")
                return {"status": "error", "message": f"未知函数: {function_name}"}
        else:
            # 如果没有工具调用，直接执行默认训练
            print_warning("AI服务未返回工具调用，执行默认训练")
            return train_yolo()
            
    except Exception as e:
        print_error(f"处理指令时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"处理指令时出错: {str(e)}"}
    
if __name__ == "__main__":
    print_success("YOLO训练助手已启动 - 请用自然语言输入指令")
    print_info("例如: \"帮我训练一个yolov8模型\" 或 \"开始模型训练\"")
    print_info("输入'退出'结束程序")
    print("-" * 50)
    
    # 命令行模式
    while True:
        instruction = input("\n请输入指令: ")
        if instruction.lower() in ['退出', 'exit', 'quit']:
            print_info("程序已退出")
            break
        
        print("-" * 50)
        result = process_instruction(instruction)
        print("-" * 50)
        print_info(f"指令处理完成: {result['message']}")
        print("-" * 50)
