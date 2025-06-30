# YOLO & CVAT 自动化系统 - 第六版 (MCP协议)

[English](README.md) | [中文](README_CN.md)

基于**模型上下文协议（MCP）**实现的综合机器学习工作流管理系统，直接集成到Claude Desktop中，通过AI对话实现YOLO模型训练、CVAT数据标注、模型部署和推理的全流程自动化。

## 概述

第六版代表了从HTTP客户端-服务器架构到**模型上下文协议（MCP）**标准的根本性架构转变。通过与Claude Desktop的直接集成，用户可以通过与Claude的自然语言对话来控制整个ML流水线，无需单独的客户端应用程序。

## 架构演进

### 之前的架构（第五版）
```
客户端 (main.py) <--HTTP--> 服务器 (server.py) <---> 各功能模块
```

### 新的MCP架构（第六版）
```
Claude Desktop <--MCP/stdio--> standard_mcp_server.py <---> 功能模块
                                        |
                                        +---> 训练 (train.py)
                                        +---> 浏览器 (browser.py)
                                        +---> CVAT API (cvat_api.py)
                                        +---> 部署 (deploy_to_cvat.py)
                                        |
                                        v
                                 Nuclio服务 <--> CVAT系统
```

## 关键改进

| 特性 | 第五版 | 第六版 |
|------|--------|--------|
| **架构** | HTTP客户端-服务器 | MCP协议 |
| **用户界面** | 独立Python客户端 | Claude Desktop集成 |
| **通信方式** | RESTful API | stdio流 |
| **自然语言** | 基于API的解析 | 原生Claude理解 |
| **安装** | 复杂的多组件 | 简单的MCP服务器配置 |
| **用户体验** | CLI菜单系统 | 对话式AI界面 |

## 核心组件

### 主模块
- **standard_mcp_server.py**: MCP服务器实现，处理所有工具注册和请求路由

### 功能模块（未改变）
- **train.py**: 使用Ultralytics的YOLO模型训练
- **browser.py**: 通过Playwright的浏览器自动化
- **cvat_api.py**: 完整的CVAT REST API封装
- **deploy_to_cvat.py**: Nuclio部署自动化
- **config.py**: 系统配置管理

### 支持文件
- **requirements_mcp.txt**: MCP特定依赖
- **deploy_templates/**: Nuclio函数模板

## 可用工具

MCP服务器向Claude暴露11个工具：

1. **hello** - 测试MCP连接
2. **train_yolo** - 异步训练YOLOv8模型
3. **query_task_status** - 检查训练任务状态
4. **list_tasks** - 列出所有训练任务
5. **browser_control** - 控制浏览器访问CVAT
6. **deploy_model** - 部署模型到Nuclio
7. **upload_data** - 上传数据到CVAT
8. **auto_annotation** - 运行AI自动标注
9. **download_dataset** - 从CVAT导出数据集
10. **convert_dataset** - 转换CVAT到YOLO格式
11. **download_and_convert** - 一键下载和转换
12. **view_annotations** - 打开标注界面

## 安装

### 前置要求
- Python 3.8+
- Docker（用于CVAT和Nuclio）
- Claude Desktop应用
- WSL（Windows用户）

### 1. 克隆仓库
```bash
git clone <repository-url>
cd YOLO_Sum/iteration-6
```

### 2. 安装依赖
```bash
pip install -r requirements_mcp.txt
python -m playwright install
```

### 3. 设置CVAT（如果尚未安装）
```bash
git clone https://github.com/opencv/cvat.git
cd cvat
docker-compose up -d
```

### 4. 在Claude Desktop中配置MCP

添加到Claude Desktop配置文件：
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yolo-cvat": {
      "command": "python",
      "args": ["C:/path/to/your/standard_mcp_server.py"],
      "cwd": "C:/path/to/your/iteration-6"
    }
  }
}
```

### 5. 配置CVAT连接
编辑 `config.py`:
```python
CVAT_URL = "http://localhost:8080"
CVAT_USERNAME = "admin"
CVAT_PASSWORD = "your_password"
```

### 6. 重启Claude Desktop
配置后重启Claude Desktop以加载MCP服务器。

## 使用示例

### 自然语言命令

直接用自然语言与Claude对话：

**训练：**
- "训练一个YOLOv8n模型10个epoch"
- "使用最新转换的数据集进行训练"
- "检查我的训练任务状态"

**数据管理：**
- "上传C:\my_data的图片到CVAT"
- "以COCO格式下载任务14的数据集"
- "将下载的数据集转换为YOLO格式"

**模型部署：**
- "部署最新训练的模型到CVAT"
- "使用我在/path/to/model.pt的自定义模型"

**标注：**
- "对任务15运行自动标注"
- "打开任务20的标注界面"

### 工作流示例

#### 完整训练流程
1. "上传我的图片到CVAT创建新任务"
2. "使用默认模型对任务进行自动标注"
3. "打开标注界面查看结果"
4. "下载并转换数据集为YOLO格式"
5. "用转换后的数据集训练YOLOv8n模型"
6. "将训练好的模型部署回CVAT"

#### 快速标注
1. "创建新任务并从我的文件夹上传数据"
2. "用最新部署的模型自动标注"
3. "展示标注结果"

## 技术实现

### MCP协议
- 使用标准Python MCP SDK
- 通过stdio流通信
- 使用线程的异步工具执行
- 实时状态更新

### 任务管理
- 在`tasks.json`中持久化任务跟踪
- 带进度监控的异步训练
- 在`mcp_server.log`中的全面日志记录

### 浏览器自动化
- 基于Playwright的浏览器控制
- 自动CVAT登录
- 支持Chrome、Edge、Firefox

## 文件结构
```
/iteration-6
  ├── standard_mcp_server.py  # MCP服务器
  ├── train.py               # 训练模块
  ├── browser.py             # 浏览器自动化
  ├── cvat_api.py           # CVAT API封装
  ├── deploy_to_cvat.py     # 部署模块
  ├── config.py             # 配置
  ├── requirements_mcp.txt   # 依赖
  ├── deploy_templates/      # Nuclio模板
  │   ├── function.yaml
  │   └── main.py
  ├── tasks.json            # 任务跟踪（生成）
  ├── runs/                 # 训练输出（生成）
  ├── downloads/            # 下载的数据集（生成）
  └── datasets/             # 转换的数据集（生成）
```

## 故障排除

### MCP连接问题
1. 检查Claude Desktop日志中的MCP错误
2. 验证配置中的Python路径
3. 确保所有依赖已安装
4. 查看`mcp_server.log`获取详细错误

### CVAT连接失败
1. 验证Docker容器正在运行：`docker ps`
2. 检查CVAT是否可访问：http://localhost:8080
3. 确认`config.py`中的凭据

### 训练失败
1. 检查GPU可用性以进行CUDA加速
2. 验证数据集路径和格式
3. 查看`runs/logs/`中的训练日志

### 浏览器自动化问题
1. 更新Playwright浏览器：`python -m playwright install`
2. 如果一个浏览器失败，尝试不同的浏览器类型
3. 检查是否有冲突的浏览器实例

## 性能提示

- 使用GPU加速进行训练和推理
- 根据可用内存调整批次大小
- 为重复部署启用模型缓存
- 为数据处理使用适当的工作进程数

## 从第五版迁移

1. 删除旧的HTTP客户端（`main.py`）和服务器（`server.py`）
2. 从`requirements_mcp.txt`安装MCP依赖
3. 在Claude Desktop中配置MCP服务器路径
4. 所有现有模块保持兼容

## 贡献

欢迎贡献！改进领域：
- 额外的模型架构支持
- 扩展的标注类型
- 性能优化
- 增强的错误处理

## 许可证

MIT许可证 - 详见LICENSE文件。

## 致谢

- 基于 [Anthropic MCP SDK](https://github.com/anthropics/mcp) 构建
- 由 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 驱动
- 集成 [CVAT](https://github.com/opencv/cvat)
- 浏览器自动化由 [Playwright](https://playwright.dev) 提供
