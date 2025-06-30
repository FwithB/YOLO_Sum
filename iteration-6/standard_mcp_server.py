#!/usr/bin/env python3
"""
standard_mcp_server.py  –  YOLO-MCP Gateway  (Threading Version)
"""

import asyncio
import io
import json
import logging
import os
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import threading
import subprocess
import sys
from playwright.async_api import async_playwright

# ─────────────────────────  MCP SDK  ──────────────────────────
from mcp.server import Server, NotificationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.server.models import InitializationOptions

# ─────────────────────────  全局设置  ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler("mcp_server.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

# 目录准备
RUNS_DIR = Path("runs")
LOGS_DIR = RUNS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TASK_DB = Path("tasks.json")
if not TASK_DB.exists():
    TASK_DB.write_text("{}", encoding="utf-8")

# 浏览器控制器
BROWSER = None
BROWSER_CONTEXT = None
BROWSER_PAGE = None
PLAYWRIGHT = None

# ───────────────────────  状态读写工具  ────────────────────────
def _save_status(task_id: str, status: str, detail: Any):
    """写入 / 更新任务状态，并返回最新条目"""
    try:
        db = json.loads(TASK_DB.read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        db = {}

    db[task_id] = {
        "status": status,
        "detail": detail,
        "updated_at": datetime.utcnow().isoformat()
    }
    TASK_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), "utf-8")
    logger.info(f"[status] {task_id} → {status}")
    return db[task_id]

# ────────────────────  线程执行训练  ──────────────────────
def run_training_thread(task_id: str, model_type: str, epochs: int, data: str):
    """在独立线程中运行训练"""
    logger.info(f"[Thread] Starting training for {task_id}")
    
    # 标记 running
    _save_status(task_id, "running", {
        "model_type": model_type,
        "epochs": epochs,
        "data": data,
        "started_at": datetime.utcnow().isoformat()
    })
    
    try:
        # 使用subprocess调用train.py避免GIL问题
        cmd = [
            sys.executable,
            "train.py",
            "--model", model_type,
            "--epochs", str(epochs),
            "--data", data
        ]
        
        log_path = LOGS_DIR / f"{task_id}.log"
        
        # 运行训练并捕获输出
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if result.returncode == 0:
            run_name = f"{model_type}_custom"
            _save_status(task_id, "done", {
                "metrics": {
                    "epochs": epochs,
                    "model": model_type
                },
                "artifacts": {
                    "weights": f"runs/detect/{run_name}/weights/best.pt",
                    "logs": str(log_path)
                },
                "message": "训练完成"
            })
        else:
            _save_status(task_id, "error", {
                "message": f"训练失败，返回码: {result.returncode}",
                "logs": str(log_path)
            })
            
    except Exception as e:
        logger.exception(f"Thread error for {task_id}")
        _save_status(task_id, "error", {
            "message": str(e),
            "logs": str(LOGS_DIR / f"{task_id}.log")
        })


# ────────────────────────  MCP server 定义  ─────────────────────
server = Server("agent-training-mcp")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """暴露给 Claude 的工具列表"""
    return [
        types.Tool(
            name="hello",
            description="测试 MCP 连接",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "名字"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="train_yolo",
            description="异步训练 YOLOv8 模型",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {"type": "string", "default": "yolov8n"},
                    "epochs":     {"type": "integer", "default": 1, "minimum": 1},
                    "data":       {"type": "string", "default": "coco128.yaml"}
                }
            }
        ),
        types.Tool(
            name="query_task_status",
            description="查询任务状态",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"]
            }
        ),
        types.Tool(
            name="list_tasks",
            description="列出所有任务",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="browser_control",
            description="控制浏览器访问CVAT",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open_cvat", "close", "navigate", "open_annotation"],
                        "description": "浏览器操作类型"
                    },
                    "url": {"type": "string", "description": "导航URL"},
                    "task_id": {"type": "string", "description": "任务ID"},
                    "job_id": {"type": "integer", "default": 1, "description": "作业ID"}
                },
                "required": ["action"]
            }
        ),
        types.Tool(
            name="deploy_model",
            description="部署模型到CVAT Nuclio",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "模型文件路径"},
                    "function_name": {"type": "string", "description": "Nuclio函数名"},
                    "force": {"type": "boolean", "default": False, "description": "强制重新部署"}
                }
            }
        ),

        types.Tool(
            name="upload_data",
            description="上传数据到CVAT创建新任务",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "数据路径（可选，不指定则使用默认路径）"},
                    "task_name": {"type": "string", "description": "任务名称（可选）"}
                },
                "required": []  # 都不是必需的
            }
        ),

        types.Tool(
            name="auto_annotation",
            description="使用AI模型对CVAT任务进行自动标注",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "CVAT任务ID"},
                    "model_name": {"type": "string", "description": "模型名称（可选，默认使用最新部署的模型）"}
                },
                "required": ["task_id"]
            }
        ),

        types.Tool(
            name="download_dataset",
            description="从CVAT下载数据集",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "CVAT任务ID"},
                    "output_path": {"type": "string", "description": "输出路径（可选）"},
                    "format": {
                        "type": "string", 
                        "default": "CVAT",
                        "enum": ["CVAT", "COCO", "COCO_KEYPOINTS", "DATUMARO", "IMAGENET", "KITTI", "CAMVID", "CITYSCAPES"],
                        "description": "导出格式"
                    },
                    "include_images": {"type": "boolean", "default": True, "description": "是否包含图像"}
                },
                "required": ["task_id"]
            }
        ),

        types.Tool(
            name="convert_dataset",
            description="将CVAT格式转换为YOLO格式",
            inputSchema={
                "type": "object",
                "properties": {
                    "cvat_path": {"type": "string", "description": "CVAT数据集路径"},
                    "output_dir": {"type": "string", "description": "输出目录（可选）"},
                    "dataset_name": {"type": "string", "description": "数据集名称（可选）"},
                    "val_split": {"type": "number", "default": 0.2, "description": "验证集比例"}
                },
                "required": ["cvat_path"]
            }
        ),

        types.Tool(
            name="download_and_convert",
            description="下载CVAT数据集并转换为YOLO格式",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "CVAT任务ID"},
                    "output_dir": {"type": "string", "description": "输出目录（可选）"},
                    "dataset_name": {"type": "string", "description": "数据集名称（可选）"},
                    "val_split": {"type": "number", "default": 0.2, "description": "验证集比例"}
                },
                "required": ["task_id"]
            }
        ),

        types.Tool(
            name="view_annotations",
            description="在浏览器中打开CVAT标注界面查看标注结果",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "CVAT任务ID"},
                    "job_id": {"type": "integer", "default": 1, "description": "作业ID"}
                },
                "required": ["task_id"]
            }
        ),

    ]

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    return []

@server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    return []

# ─────────────────────────  工具入口  ──────────────────────────
@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info(f"Tool called: {name}  args: {arguments}")

    # ① hello
    if name == "hello":
        user_name = arguments.get("name", "World")
        return [{"type": "text", "text": f"你好, {user_name}! MCP 连接正常 🎉"}]

    # ② train_yolo
    elif name == "train_yolo":
        model_type = arguments.get("model_type", "yolov8n")
        epochs     = int(arguments.get("epochs", 1))
        data       = arguments.get("data", "coco128.yaml")

        if epochs < 1:
            return [{"type": "text", "text": "❌ epochs 必须 ≥ 1"}]

        task_id = f"T{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        _save_status(task_id, "queued", {
            "model_type": model_type, "epochs": epochs, "data": data,
            "created_at": datetime.utcnow().isoformat()
        })

        # 启动训练线程
        thread = threading.Thread(
            target=run_training_thread,
            args=(task_id, model_type, epochs, data)
        )
        thread.daemon = True
        thread.start()

        return [{
            "type": "text",
            "text": (
                f"✅ 新训练任务已创建\n\n"
                f"🆔 {task_id}\n"
                f"模型: {model_type} | 轮数: {epochs} | 数据: {data}\n"
                f"当前状态: queued\n\n"
                f"用 query_task_status(task_id=\"{task_id}\") 查询进度。"
            )
        }]

# ③ query_task_status
    elif name == "query_task_status":
        task_id = arguments.get("task_id")
        if not task_id:
            return [{"type": "text", "text": "❌ 请提供 task_id"}]

        try:
            db = json.loads(TASK_DB.read_text("utf-8"))
            task = db.get(task_id)
        except Exception as e:
            logger.error(e)
            task = None

        if not task:
            return [{"type": "text", "text": f"❓ 未找到任务 {task_id}"}]

        status = task["status"]
        detail = task.get("detail", {})
        emoji  = {"queued":"⏳","running":"🏃","done":"✅","error":"❌"}.get(status, "❓")

        msg = f"{emoji} 任务 {task_id}\n状态: {status}\n"
        if status == "done" and isinstance(detail, dict):
            art = detail.get("artifacts", {})
            msg += f"权重: {art.get('weights')}\n日志: {art.get('logs')}\n"
        if status == "error":
            msg += f"错误: {detail.get('message')}\n"

        msg += f"更新时间: {task.get('updated_at')}"
        return [{"type": "text", "text": msg}]

    # ④ list_tasks
    elif name == "list_tasks":
        db = json.loads(TASK_DB.read_text("utf-8"))
        if not db:
            return [{"type": "text", "text": "📭 暂无任务"}]

        lines = []
        for tid, t in sorted(db.items(), reverse=True):
            em = {"queued":"⏳","running":"🏃","done":"✅","error":"❌"}.get(t["status"], "❓")
            lines.append(f"{em} {tid}  -  {t['status']}")
        return [{"type": "text", "text": "📋 任务列表：\n" + "\n".join(lines)}]

    # ⑤ browser_control
    elif name == "browser_control":
        action = arguments.get("action")
        
        if action == "open_cvat":
            return await handle_open_cvat()
        elif action == "close":
            return await handle_close_browser()
        elif action == "navigate":
            url = arguments.get("url")
            if not url:
                return [{"type": "text", "text": "❌ 导航需要提供URL"}]
            return await handle_navigate(url)
        elif action == "open_annotation":
            task_id = arguments.get("task_id")
            job_id = arguments.get("job_id", 1)
            if not task_id:
                return [{"type": "text", "text": "❌ 需要提供task_id"}]
            return await handle_open_annotation(task_id, job_id)
        else:
            return [{"type": "text", "text": f"❌ 未知操作: {action}"}]
    # ⑥ deploy_model
# ────────────────────── ⑥ deploy_model ──────────────────────
    elif name == "deploy_model":
        model_path = arguments.get("model_path")
        function_name = arguments.get("function_name")
        force = arguments.get("force", False)
        
        # 验证用户指定的路径
        if model_path and not Path(model_path).exists():
            logger.warning(f"用户指定的路径不存在: {model_path}")
            model_path = None  # 清空，让它自动查找
        
        if not model_path:
            # 先尝试导入函数看看
            try:
                from deploy_to_cvat import get_latest_model_from_runs
                model_path = get_latest_model_from_runs()
                logger.info(f"get_latest_model_from_runs返回: {model_path}")
            except Exception as e:
                logger.error(f"导入失败: {e}")
                model_path = None
            
            # 如果没找到或路径不存在，用备用方法
            if not model_path or not Path(model_path).exists():
                logger.info("使用备用查找方法")
                runs_dir = Path("runs/detect")
                if runs_dir.exists():
                    valid_dirs = []
                    for d in runs_dir.iterdir():
                        if d.is_dir() and (d / "weights" / "best.pt").exists():
                            valid_dirs.append(d)
                    if valid_dirs:
                        latest_dir = max(valid_dirs, key=lambda d: d.stat().st_mtime)
                        model_path = str(latest_dir / "weights" / "best.pt")
                        logger.info(f"找到模型: {model_path}")
            
            if not model_path:
                return [{"type": "text", "text": "❌ 未找到可部署的模型"}]
        
        # 保存实际使用的路径
        actual_model_path = model_path
        
        cmd = ['python3', 'deploy_to_cvat.py']
        if model_path:
            cmd.append(model_path)
        #if function_name:
        #    cmd.extend(['--name', function_name])
        if force:
            cmd.append('--force')
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=1200)
            logger.info(f"返回码: {result.returncode}")
            logger.info(f"完整输出: {result.stdout}")
            logger.info(f"错误输出: {result.stderr}")
            
            if result.returncode == 0:
                # JSON在stderr中
                if result.stderr:
                    try:
                        deploy_result = json.loads(result.stderr.strip())
                        if deploy_result.get("status") == "error":
                            return [{"type": "text", "text": f"❌ 部署失败: {deploy_result.get('message', '未知错误')}"}]
                        return [{"type": "text", "text": f"✅ 部署成功\n函数名: {deploy_result.get('function_name', 'N/A')}\n模型路径: {actual_model_path}"}]
                    except json.JSONDecodeError:
                        pass
                return [{"type": "text", "text": f"✅ 部署完成\n模型路径: {actual_model_path}"}]
            return [{"type": "text", "text": f"❌ 部署失败: {result.stderr or result.stdout}"}]
        except Exception as e:
            logger.exception("部署异常")
            return [{"type": "text", "text": f"❌ 部署出错: {str(e)}"}]
        

    elif name == "upload_data":
        data_path = arguments.get("data_path")
        
        # 自动转换 Windows 路径到 WSL 路径
        if data_path:
            # 处理各种 Windows 路径格式
            if data_path.startswith("C:\\") or data_path.startswith("C:/") or data_path.startswith("C:"):
                # C:\path → /mnt/c/path
                data_path = data_path.replace("C:\\", "/mnt/c/")
                data_path = data_path.replace("C:/", "/mnt/c/")
                data_path = data_path.replace("C:", "/mnt/c")
                # 统一使用正斜杠
                data_path = data_path.replace("\\", "/")
                logger.info(f"转换 Windows 路径到 WSL: {data_path}")
        
        task_name = arguments.get("task_name")
        
        try:
            from cvat_api import upload_data_to_cvat
            
            result = upload_data_to_cvat(
                data_path=data_path,
                task_name=task_name
            )
            
            if result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 数据上传成功!\n任务名: {task_name or '自动生成'}\n任务ID: {result['task_id']}"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 上传失败: {result['message']}"
                }]
                
        except Exception as e:
            logger.exception("上传数据异常")
            return [{"type": "text", "text": f"❌ 上传出错: {str(e)}"}]
            
    elif name == "auto_annotation":
        task_id = arguments.get("task_id")
        model_name = arguments.get("model_name")
        
        try:
            from cvat_api import run_auto_annotation
            
            result = run_auto_annotation(
                task_id=task_id,
                model_name=model_name
            )
            
            if result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 自动标注成功!\n任务ID: {task_id}\n使用模型: {model_name or '默认YOLO模型'}\n请求ID: {result.get('request_id', 'N/A')}"
                }]
            elif result["status"] == "warning":
                return [{
                    "type": "text",
                    "text": f"⚠️ 自动标注已提交但有警告\n任务ID: {task_id}\n警告信息: {result.get('message', 'N/A')}\n异常: {result.get('exception', 'N/A')}"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 自动标注失败: {result['message']}"
                }]
                
        except Exception as e:
            logger.exception("自动标注异常")
            return [{"type": "text", "text": f"❌ 自动标注出错: {str(e)}"}] 
        

    elif name == "download_dataset":
        task_id = arguments.get("task_id")
        output_path = arguments.get("output_path")
        format = arguments.get("format", "CVAT")
        include_images = arguments.get("include_images", True)
        
        try:
            from cvat_api import download_dataset
            
            result = download_dataset(
                task_id=task_id,
                output_path=output_path,
                format=format,
                include_images=include_images
            )
            
            if result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 数据集下载成功!\n"
                        f"任务ID: {task_id}\n"
                        f"格式: {result['format']}\n"
                        f"输出路径: {result['output_path']}\n"
                        f"ZIP文件: {result['zip_path']}"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 下载失败: {result['message']}"
                }]
                
        except Exception as e:
            logger.exception("下载数据集异常")  
            return [{"type": "text", "text": f"❌ 下载出错: {str(e)}"}]

    elif name == "convert_dataset":
        cvat_path = arguments.get("cvat_path")
        output_dir = arguments.get("output_dir")
        dataset_name = arguments.get("dataset_name")
        val_split = arguments.get("val_split", 0.2)
        
        try:
            from cvat_api import convert_cvat_to_yolo
            
            result = convert_cvat_to_yolo(
                cvat_path=cvat_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                val_split=val_split
            )
            
            if result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 转换成功!\n"
                        f"输出目录: {result['dataset_dir']}\n"
                        f"YAML配置: {result['yaml_path']}\n"
                        f"训练集: {result['train_count']}张\n"
                        f"验证集: {result['val_count']}张"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 转换失败: {result['message']}"
                }]
                
        except Exception as e:
            logger.exception("转换数据集异常")
            return [{"type": "text", "text": f"❌ 转换出错: {str(e)}"}]
        
    elif name == "download_and_convert":
        task_id = arguments.get("task_id")
        output_dir = arguments.get("output_dir")
        dataset_name = arguments.get("dataset_name")
        val_split = arguments.get("val_split", 0.2)
        
        try:
            from cvat_api import download_dataset, convert_cvat_to_yolo
            
            # 先下载
            download_result = download_dataset(task_id=task_id)
            if download_result["status"] != "success":
                return [{
                    "type": "text",
                    "text": f"❌ 下载失败: {download_result['message']}"
                }]
            
            # 再转换
            convert_result = convert_cvat_to_yolo(
                cvat_path=download_result["output_path"],
                output_dir=output_dir,
                dataset_name=dataset_name or f"task_{task_id}_yolo",
                val_split=val_split
            )
            
            if convert_result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 下载并转换成功!\n"
                        f"YOLO数据集: {convert_result['dataset_dir']}\n"
                        f"YAML配置: {convert_result['yaml_path']}\n"
                        f"训练集: {convert_result['train_count']}张\n"
                        f"验证集: {convert_result['val_count']}张"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 转换失败: {convert_result['message']}"
                }]
                
        except Exception as e:
            logger.exception("下载转换异常")
            return [{"type": "text", "text": f"❌ 出错: {str(e)}"}]

    elif name == "view_annotations":
        task_id = arguments.get("task_id")
        job_id = arguments.get("job_id", 1)
        
        try:
            from browser import open_annotation_interface_sync
            
            result = open_annotation_interface_sync(task_id, job_id)
            
            if result["status"] == "success":
                return [{
                    "type": "text",
                    "text": f"✅ 已打开标注界面\n任务ID: {task_id}\nJob ID: {job_id}"
                }]
            else:
                return [{
                    "type": "text",
                    "text": f"❌ 打开失败: {result['message']}"
                }]
                
        except Exception as e:
            logger.exception("打开标注界面异常")
            return [{"type": "text", "text": f"❌ 出错: {str(e)}"}]
    # 未识别
    return [{"type": "text", "text": f"❌ 未知工具 {name}"}]

# ────────────────────── 浏览器控制函数 ──────────────────────
async def handle_open_cvat():
   """打开CVAT（新窗口）"""
   global PLAYWRIGHT, BROWSER
   
   try:
       # 初始化
       if not PLAYWRIGHT:
           PLAYWRIGHT = await async_playwright().start()
       
       if not BROWSER:
           BROWSER = await PLAYWRIGHT.chromium.launch(headless=False)
       
       # 创建新的独立窗口
       new_context = await BROWSER.new_context()
       new_page = await new_context.new_page()
       
       await new_page.goto("http://localhost:8080/auth/login")
       await new_page.wait_for_load_state('networkidle')
       await asyncio.sleep(2)
       
       try:
           # 登录逻辑保持不变
           selectors = [
               'input[placeholder="Email or username"]',
               'input[type="text"]',
               'input[name="username"]',
               'input#username',
               '.ant-input'
           ]
           
           username_input = None
           for selector in selectors:
               try:
                   username_input = await new_page.query_selector(selector)
                   if username_input:
                       break
               except:
                   continue
           
           if username_input:
               await username_input.click()
               await username_input.fill("admin")
               await username_input.press("Enter")
               
               await asyncio.sleep(2)
               password_input = await new_page.query_selector('input[type="password"]')
               if password_input:
                   await password_input.click()
                   await password_input.fill("Yyh277132984")
                   await password_input.press("Enter")
                   
               return [{"type": "text", "text": "✅ 已在新窗口打开CVAT并自动登录"}]
               
       except Exception as e:
           logger.warning(f"自动登录失败: {e}")
           return [{"type": "text", "text": "⚠️ 已打开新窗口但自动登录失败"}]
       
   except Exception as e:
       logger.error(f"打开CVAT失败: {e}")
       return [{"type": "text", "text": f"❌ 打开CVAT失败: {str(e)}"}]

async def handle_close_browser():
    """关闭浏览器"""
    global BROWSER
    if BROWSER:
        await BROWSER.close()
        BROWSER = None
        return [{"type": "text", "text": "✅ 浏览器已关闭"}]
    return [{"type": "text", "text": "ℹ️ 没有打开的浏览器"}]

async def handle_navigate(url: str):
    """导航到URL"""
    if not BROWSER_PAGE:
        return [{"type": "text", "text": "❌ 请先打开浏览器"}]
    
    try:
        await BROWSER_PAGE.goto(url)
        return [{"type": "text", "text": f"✅ 已导航到: {url}"}]
    except Exception as e:
        return [{"type": "text", "text": f"❌ 导航失败: {str(e)}"}]

async def handle_open_annotation(task_id: str, job_id: int):
    """打开标注界面"""
    if not BROWSER_PAGE:
        result = await handle_open_cvat()
        if "❌" in result[0]["text"]:
            return result
    
    try:
        url = f"http://localhost:8080/tasks/{task_id}/jobs/{job_id}"
        await BROWSER_PAGE.goto(url)
        await BROWSER_PAGE.wait_for_selector('.cvat-canvas-container', timeout=20000)
        return [{"type": "text", "text": f"✅ 已打开任务{task_id}的标注界面"}]
    except Exception as e:
        return [{"type": "text", "text": f"❌ 打开标注界面失败: {str(e)}"}]


# ───────────────────────────  启动  ────────────────────────────
async def main():
    logger.info("🚀 Starting MCP server …")
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(
            r, w,
            InitializationOptions(
                server_name="agent-training-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())

