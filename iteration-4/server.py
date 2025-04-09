"""
MCP服务器 - 接收训练请求和浏览器控制请求
"""

from flask import Flask, request, jsonify
import json
import multiprocessing
import subprocess
import os
import sys
import logging

# 引入训练和浏览器控制模块
from train import train_yolo
from browser import process_browser_request

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger('mcp_server')

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_endpoint():
    """处理训练请求的接口"""
    try:
        # 解析JSON请求体
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "缺少JSON请求体"}), 400
        
        logger.info(f"Received training request: {json.dumps(data)}")
        
        # 从JSON中获取训练参数（如果没有，设置默认值）
        model_type = data.get('model_type', 'yolov8n')
        epochs = data.get('epochs', 1)
        dataset = data.get('data', 'coco128.yaml')
        
        # 调用训练函数
        result = train_yolo(model_type, epochs, dataset)
        
        logger.info(f"Training result: {json.dumps(result)}")
        
        # 将训练结果以JSON形式返回
        return jsonify(result), 200
    
    except Exception as e:
        logger.exception(f"Error handling training request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/browser', methods=['POST'])
def browser_endpoint():
    """处理浏览器控制请求的接口"""
    try:
        # 解析JSON请求体
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "缺少JSON请求体"}), 400
        
        logger.info(f"Received browser request: {json.dumps(data)}")
        
        # 处理浏览器请求
        result = process_browser_request(data)
        
        logger.info(f"Browser control result: {json.dumps(result)}")
        
        # 将结果以JSON形式返回
        return jsonify(result), 200
    
    except Exception as e:
        logger.exception(f"Error handling browser request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "success", "message": "Server is running"}), 200

@app.route('/', methods=['GET'])
def index():
    """根路径，提供服务信息"""
    return jsonify({
        "name": "YOLO MCP Server with Browser Control",
        "endpoints": [
            {"path": "/train", "method": "POST", "description": "训练YOLO模型"},
            {"path": "/browser", "method": "POST", "description": "控制浏览器"},
            {"path": "/health", "method": "GET", "description": "健康检查"}
        ],
        "status": "running"
    }), 200

if __name__ == '__main__':
    # 确保在Windows上正确运行多进程
    multiprocessing.freeze_support()
    
    # 获取端口（默认5000）
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting MCP server on port {port}")
    print(f"启动MCP服务器，监听端口 {port}...")
    
    # 在0.0.0.0上监听指定端口
    app.run(host='0.0.0.0', port=port, debug=False)
