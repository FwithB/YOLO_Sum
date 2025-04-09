"""
MCP客户端 - 处理用户交互、解析指令并发送请求
"""

import requests
import json
import os
import sys
import logging
from openai import OpenAI  # 需要安装: pip install openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_client.log')
    ]
)
logger = logging.getLogger('mcp_client')

# 设置API密钥 - 从环境变量获取或直接设置
API_KEY = "bcf6fa84c7c4468ea02a021e62fdec97"
if not API_KEY:
    logger.warning("未设置OPENROUTER_API_KEY环境变量，LLM解析功能可能无法正常工作")
    logger.warning("请设置环境变量: export OPENROUTER_API_KEY=你的密钥")

# 服务器配置
SERVER_URL = "http://localhost:5000"  # 可以修改为实际服务器地址
TRAIN_ENDPOINT = f"{SERVER_URL}/train"
BROWSER_ENDPOINT = f"{SERVER_URL}/browser"

# OpenRouter基础URL
BASE_URL = "http://195.179.229.119/gpt"

# 初始化OpenAI客户端 (OpenRouter兼容OpenAI API)
client = None
if API_KEY:
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )
    except Exception as e:
        logger.error(f"初始化OpenAI客户端失败: {str(e)}")

def parse_instruction_with_llm(user_query):
    """
    使用LLM解析用户的自然语言指令，转换为结构化参数
    
    Args:
        user_query: 用户输入的自然语言指令
        
    Returns:
        dict: 包含解析后的参数和操作类型
    """
    import re
    # 如果客户端未初始化，返回默认参数
    if client is None:
        logger.error("LLM客户端未初始化，无法解析指令")
        return {"operation": "unknown", "params": {}}
    
    try:
        # 定义函数调用格式 - 保留原始定义，虽然不再直接使用
        functions = [
            {
                "name": "train_yolo",
                "description": "训练YOLOv8模型的函数",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "YOLO模型类型，例如yolov8n、yolov8s、yolov8m、yolov8l、yolov8x等"
                        },
                        "epochs": {
                            "type": "integer",
                            "description": "训练轮数，默认为1"
                        },
                        "data": {
                            "type": "string",
                            "description": "数据集配置文件路径，如coco128.yaml"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "control_browser",
                "description": "控制浏览器的函数，如打开网页、导航到URL等",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "浏览器动作，如initialize(初始化), navigate(导航), open_cvat(打开CVAT), close(关闭)",
                            "enum": ["initialize", "navigate", "open_cvat", "login_cvat", "create_project", "close", "take_screenshot"]
                        },
                        "url": {
                            "type": "string",
                            "description": "要导航到的URL，仅在action为navigate时使用"
                        },
                        "browser_type": {
                            "type": "string",
                            "description": "浏览器类型，如chromium, firefox, webkit",
                            "enum": ["chromium", "firefox", "webkit"]
                        },
                        "username": {
                            "type": "string",
                            "description": "登录用户名，仅在action为login_cvat时使用"
                        },
                        "password": {
                            "type": "string",
                            "description": "登录密码，仅在action为login_cvat时使用"
                        },
                        "project_name": {
                            "type": "string",
                            "description": "项目名称，仅在action为create_project时使用"
                        }
                    },
                    "required": ["action"]
                }
            }
        ]
        
        logger.info(f"发送指令到LLM进行解析: '{user_query}'")
        
        # 使用第三方API而非OpenAI客户端
        url = f"{BASE_URL}/api.php"
        system_prompt = """你是一个专业的计算机视觉助手，负责解析用户指令。
                    
                    当用户想要训练模型时，使用train_yolo函数。
                    当用户想要控制浏览器时，使用control_browser函数。
                    
                    特别地，如果用户提到：
                    - "打开浏览器"、"启动浏览器"、"打开网页"等，应该使用control_browser函数并设置适当的action
                    - 如果明确提到"CVAT"或"打开CVAT"，应该使用control_browser函数，并将action设为"open_cvat"
                    
                    如果不确定用户意图，优先使用最可能的函数，并设置适当的默认参数。"""
        
        prompt = f"{system_prompt}\n\n用户指令: '{user_query}'\n\n请返回以下JSON格式：\n{{\"operation\": \"train或browser\", \"params\": {{相应的参数}}}}。如果是训练操作，params应包含model_type, epochs, data；如果是浏览器操作，params应包含action和其他必要参数。"
        
        payload = {
            "prompt": prompt,
            "api_key": API_KEY,
            "model": "gpt-3.5-turbo"
        }
        
        # 记录请求信息
        logger.info(f"使用第三方API解析指令: {url}")
        
        # 发送请求
        response = requests.get(url, params=payload)
        response.raise_for_status()
        data = response.json()
        
        # 记录原始响应
        logger.info(f"API原始响应: {json.dumps(data)[:200]}...")
        
        # 提取内容
        content = ""
        if "content" in data:
            content = data["content"]
        
        logger.info(f"LLM返回内容: {content}")
        
        # 尝试解析返回的JSON
        try:
            parsed_result = json.loads(content)
            
            # 验证解析结果是否符合预期格式
            if "operation" in parsed_result and "params" in parsed_result:
                operation = parsed_result["operation"]
                params = parsed_result["params"]
                
                # 根据操作类型验证参数
                if operation == "train":
                    # 确保所有必要参数都存在，设置默认值
                    default_params = {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
                    for key, default_value in default_params.items():
                        if key not in params or params[key] is None:
                            params[key] = default_value
                    
                    # 确保epochs是整数
                    try:
                        params["epochs"] = int(params["epochs"])
                    except (ValueError, TypeError):
                        params["epochs"] = 1
                
                elif operation == "browser":
                    # 确保action存在
                    if "action" not in params:
                        params["action"] = "open_cvat"  # 默认操作
                
                else:
                    logger.warning(f"未知操作类型: {operation}")
                    operation = "unknown"
                
                return {"operation": operation, "params": params}
            else:
                logger.warning("返回的JSON不符合预期格式")
        except json.JSONDecodeError:
            logger.warning("无法解析返回内容为JSON")
        
        # 如果解析失败，尝试根据关键词判断
        lower_query = user_query.lower()
        if any(kw in lower_query for kw in ["浏览器", "browser", "网页", "cvat"]):
            return {
                "operation": "browser", 
                "params": {"action": "open_cvat"}
            }
        else:
            return {
                "operation": "train", 
                "params": {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
            }
            
    except Exception as e:
        logger.exception(f"解析指令时出现异常: {str(e)}")
        # 出错时返回空结果
        return {"operation": "unknown", "params": {}}
    
    
def send_train_request(params, max_retries=2):
    """
    向服务器发送训练请求
    
    Args:
        params: 训练参数字典
        max_retries: 最大重试次数
        
    Returns:
        dict: 服务器返回的结果
    """
    logger.info(f"准备向服务器 {TRAIN_ENDPOINT} 发送训练请求: {params}")
    
    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"发送请求 (尝试 {retries+1}/{max_retries+1})...")
            response = requests.post(TRAIN_ENDPOINT, json=params, timeout=180)
            
            # 检查响应状态码
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("请求成功，服务器返回结果")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"无法解析服务器响应为JSON: {response.text}")
                    return {
                        "status": "error",
                        "message": "无法解析服务器响应",
                        "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    }
            else:
                logger.error(f"服务器返回错误状态码: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"服务端错误，状态码: {response.status_code}",
                    "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                }
                
        except requests.exceptions.Timeout:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"请求超时，将在5秒后重试...")
                import time
                time.sleep(5)
            else:
                logger.error("请求超时，已达到最大重试次数")
                return {
                    "status": "error",
                    "message": "请求超时",
                    "details": f"服务器在{180}秒内没有响应"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error(f"连接错误：无法连接到服务器 {TRAIN_ENDPOINT}")
            return {
                "status": "error",
                "message": "无法连接到服务器",
                "details": f"请确保服务器正在运行并且可以通过 {TRAIN_ENDPOINT} 访问"
            }
            
        except Exception as e:
            logger.exception(f"请求过程中发生未预期的错误: {str(e)}")
            return {
                "status": "error",
                "message": f"请求异常: {str(e)}",
                "details": "查看日志文件获取更多信息"
            }

def send_browser_request(params, max_retries=2):
    """
    向服务器发送浏览器控制请求
    
    Args:
        params: 浏览器控制参数字典
        max_retries: 最大重试次数
        
    Returns:
        dict: 服务器返回的结果
    """
    logger.info(f"准备向服务器 {BROWSER_ENDPOINT} 发送浏览器控制请求: {params}")
    
    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"发送请求 (尝试 {retries+1}/{max_retries+1})...")
            response = requests.post(BROWSER_ENDPOINT, json=params, timeout=30)
            
            # 检查响应状态码
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("请求成功，服务器返回结果")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"无法解析服务器响应为JSON: {response.text}")
                    return {
                        "status": "error",
                        "message": "无法解析服务器响应",
                        "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    }
            else:
                logger.error(f"服务器返回错误状态码: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"服务端错误，状态码: {response.status_code}",
                    "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                }
                
        except requests.exceptions.Timeout:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"请求超时，将在5秒后重试...")
                import time
                time.sleep(5)
            else:
                logger.error("请求超时，已达到最大重试次数")
                return {
                    "status": "error",
                    "message": "请求超时",
                    "details": f"服务器在30秒内没有响应"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error(f"连接错误：无法连接到服务器 {BROWSER_ENDPOINT}")
            return {
                "status": "error",
                "message": "无法连接到服务器",
                "details": f"请确保服务器正在运行并且可以通过 {BROWSER_ENDPOINT} 访问"
            }
            
        except Exception as e:
            logger.exception(f"请求过程中发生未预期的错误: {str(e)}")
            return {
                "status": "error",
                "message": f"请求异常: {str(e)}",
                "details": "查看日志文件获取更多信息"
            }

def validate_input(prompt, validator=None, default=None):
    """
    通用输入验证函数
    
    Args:
        prompt: 提示用户的文本
        validator: 验证函数，返回True/False
        default: 默认值
        
    Returns:
        验证通过的用户输入或默认值
    """
    while True:
        value = input(prompt).strip()
        
        # 如果用户未输入且有默认值，返回默认值
        if not value and default is not None:
            return default
            
        # 如果没有验证函数或验证通过，返回值
        if validator is None or validator(value):
            return value
            
        # 否则提示错误并重新请求输入
        print("输入无效，请重新输入")

def process_train_operation(params):
    """处理训练操作"""
    print("\n解析结果 (训练操作):")
    print(f"- 模型类型: {params.get('model_type', 'yolov8n')}")
    print(f"- 训练轮数: {params.get('epochs', 1)}")
    print(f"- 数据集: {params.get('data', 'coco128.yaml')}")
    
    confirm = validate_input("\n确认开始训练? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'n')
                          
    if confirm.lower() in ['y', 'yes']:
        result = send_train_request(params)
        print("\n服务器返回结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    else:
        print("已取消训练请求")
        return {"status": "canceled", "message": "用户取消了操作"}

def process_browser_operation(params):
    """处理浏览器操作"""
    action = params.get('action', 'unknown')
    
    print("\n解析结果 (浏览器操作):")
    print(f"- 动作: {action}")
    
    if action == 'navigate' and 'url' in params:
        print(f"- URL: {params['url']}")
    
    if action == 'open_cvat':
        browser_type = params.get("browser_type", "chromium")  # 默认使用Chrome
        params["browser_type"] = browser_type  # 确保参数中包含browser_type
        print(f"- 将使用{browser_type}浏览器打开并访问CVAT")
        
    
    confirm = validate_input("\n确认执行浏览器操作? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'y')
                          
    if confirm.lower() in ['y', 'yes']:
        result = send_browser_request(params)
        print("\n服务器返回结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    else:
        print("已取消浏览器操作")
        return {"status": "canceled", "message": "用户取消了操作"}

def test_server_connection():
    """测试服务器连接"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"服务器连接测试成功: {SERVER_URL}")
            return True
        else:
            logger.warning(f"服务器返回非200状态码: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"服务器连接测试失败: {str(e)}")
        return False

def print_colored(text, color_code):
    """使用ANSI颜色代码打印彩色文本"""
    print(f"\033[{color_code}m{text}\033[0m")

def print_info(text):
    print_colored(f"[信息] {text}", "34;1")  # 蓝色加粗

def print_success(text):
    print_colored(f"[成功] {text}", "32;1")  # 绿色加粗

def print_error(text):
    print_colored(f"[错误] {text}", "31;1")  # 红色加粗

def print_warning(text):
    print_colored(f"[警告] {text}", "33;1")  # 黄色加粗

def main():
    """主函数，处理用户交互和请求发送"""
    print("="*50)
    print_success("YOLO训练与浏览器控制系统 - MCP客户端")
    print("="*50)
    
    # 测试服务器连接
    if not test_server_connection():
        print_error(f"无法连接到服务器 {SERVER_URL}")
        print_info("请确认服务器已启动，并且可以通过网络访问")
        should_continue = validate_input("是否继续? (y/n): ", 
                                      lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                                      'n')
        if should_continue.lower() not in ['y', 'yes']:
            return
    
    # 检查API密钥
    if not API_KEY:
        print_warning("未设置OPENROUTER_API_KEY环境变量")
        print_info("自然语言指令解析功能将不可用，只能使用手动参数模式")
    
    while True:
        print("\n请选择操作：")
        print("1. 使用自然语言指令控制系统")
        print("2. 训练YOLO模型 (手动参数)")
        print("3. 控制浏览器 (手动参数)")
        print("4. 退出程序")
        
        choice = validate_input("请输入选项 [1-4]: ", lambda x: x in ['1', '2', '3', '4'])
        
        if choice == '1':
            # 自然语言指令模式
            if not API_KEY:
                print_error("未设置API密钥，无法使用自然语言指令")
                continue
                
            user_query = validate_input("\n请输入您的指令 (例如：'训练模型' 或 '打开浏览器访问CVAT'): ", 
                                      lambda x: len(x) > 0, 
                                      None)
            
            print_info("正在解析您的指令...")
            parsed = parse_instruction_with_llm(user_query)
            
            if parsed["operation"] == "train":
                process_train_operation(parsed["params"])
                
            elif parsed["operation"] == "browser":
                process_browser_operation(parsed["params"])
                
            else:
                print_error("无法解析指令，请尝试更明确的表述或使用手动模式")
                
        elif choice == '2':
            # 训练模型 (手动参数)
            print("\n请输入训练参数:")
            model_type = validate_input("模型类型 (默认 yolov8n): ", None, "yolov8n")
            
            epochs = validate_input("训练轮数 (默认 1): ", 
                                 lambda x: x.isdigit() and int(x) > 0, 
                                 "1")
            epochs = int(epochs)
            
            data = validate_input("数据集配置 (默认 coco128.yaml): ", None, "coco128.yaml")
            
            params = {"model_type": model_type, "epochs": epochs, "data": data}
            process_train_operation(params)
                
        elif choice == '3':
            # 控制浏览器 (手动参数)
            print("\n请选择浏览器操作:")
            print("1. 打开浏览器并访问CVAT")
            print("2. 关闭浏览器")
            print("3. 导航到指定URL")
            print("4. 登录CVAT")
            print("5. 创建CVAT项目")
            print("6. 强制重启浏览器") # 新选项
            
            browser_choice = validate_input("请输入选项 [1-6]: ", 
                                    lambda x: x in ['1', '2', '3', '4', '5', '6'])
            
            params = {}
            
            if browser_choice == '1':
                params = {"action": "open_cvat"}
                
            elif browser_choice == '2':
                params = {"action": "close"}
                
            elif browser_choice == '3':
                url = validate_input("请输入URL: ", lambda x: x.startswith("http"), None)
                params = {"action": "navigate", "url": url}
                
            elif browser_choice == '4':
                username = validate_input("用户名 (默认 admin): ", None, "admin")
                password = validate_input("密码 (默认 admin): ", None, "admin")
                params = {"action": "login_cvat", "username": username, "password": password}
                
            elif browser_choice == '5':
                name = validate_input("项目名称: ", lambda x: len(x) > 0, None)
                params = {"action": "create_project", "name": name}
                        # 处理新选项
            # 处理选项
            if browser_choice == '6':
                browser_type = validate_input("浏览器类型 (1: Chrome, 2: Edge, 3: Firefox): ", 
                                        lambda x: x in ['1', '2', '3'], '1')
                
                browser_map = {'1': 'chromium', '2': 'msedge', '3': 'firefox'}
                selected_browser = browser_map[browser_type]
                
                params = {
                    "action": "initialize", 
                    "browser_type": selected_browser,
                    "force_new": True,
                    "headless": False
                }
            
            process_browser_operation(params)
                
        elif choice == '4':
            # 退出程序
            print_success("感谢使用YOLO训练与浏览器控制系统，再见！")
            logger.info("用户选择退出程序")
            return
            
        else:
            print_error("无效选项，请重新选择")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        logger.info("程序被用户中断(KeyboardInterrupt)")
    except Exception as e:
        print_error(f"程序运行出错: {str(e)}")
        logger.exception("程序意外终止")
        sys.exit(1)