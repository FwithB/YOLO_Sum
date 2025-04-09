# YOLO-Browser-MCP

基于MCP架构的YOLO训练与浏览器控制系统，支持通过自然语言指令控制模型训练和浏览器操作。

## 功能特点

- 自然语言控制：使用LLM解析用户指令，无需记忆命令格式
- MCP架构：客户端-服务器分离，支持远程控制
- YOLO训练：支持YOLOv8各种模型的训练与参数配置
- 浏览器控制：使用Playwright实现浏览器自动化，支持打开CVAT等操作
- 日志系统：完整的日志记录，方便排查问题

## 系统架构

```
客户端 (main.py) <---> 服务器 (server.py)
                           |
                           +---> 训练模块 (train.py)
                           |
                           +---> 浏览器模块 (browser.py)
```

## 安装步骤

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/YOLO-Browser-MCP.git
cd YOLO-Browser-MCP
```

2. 安装依赖:
```bash
pip install -r requirements.txt
python -m playwright install
```

3. 配置API密钥:
在`main.py`中设置`API_KEY`变量或设置环境变量:
```bash
export OPENROUTER_API_KEY=你的API密钥
```

## 使用方法

### 启动服务器

```bash
python server.py
```

### 运行客户端

```bash
python main.py
```

### 使用示例

#### 自然语言指令示例

- 训练模型: "训练一个yolov8n模型识别猫狗，跑5轮"
- 打开CVAT: "打开浏览器，访问CVAT"
- 创建项目: "在CVAT中创建一个新项目"

#### 手动模式操作

客户端提供菜单驱动的手动操作模式，可以精确控制训练参数和浏览器行为。

## CVAT操作说明

本系统默认CVAT通过Docker在本地运行，访问地址为`http://localhost:8080`。确保:

1. Docker已启动
2. CVAT容器正在运行
3. Nuclio服务可用(用于自动标注)

## 文件说明

- `main.py`: MCP客户端，处理用户交互和指令解析
- `server.py`: MCP服务器，接收请求并调用相应模块
- `train.py`: YOLO训练模块，负责模型训练
- `browser.py`: 浏览器控制模块，使用Playwright实现浏览器自动化

## 故障排除

- 确保服务器在客户端可访问的地址运行
- 检查API密钥是否正确设置
- 验证Docker和CVAT服务是否正常运行
- 查看日志文件获取详细错误信息

## 注意事项

- 浏览器控制会在物理显示器上打开浏览器窗口，确保有图形界面环境
- 首次使用时，Playwright会自动下载浏览器驱动
- CVAT默认账号为admin，密码为admin
