# YOLO & CVAT Automation System - Iteration 6 (MCP Protocol)

[English](README.md) | [中文](README_CN.md)

A comprehensive machine learning workflow management system implementing the **Model Context Protocol (MCP)**, directly integrated with Claude Desktop for seamless AI-assisted automation of YOLO model training, CVAT data annotation, model deployment, and inference.

## Overview

Iteration 6 represents a fundamental architectural shift from HTTP-based client-server to the **Model Context Protocol (MCP)** standard. This enables direct integration with Claude Desktop, allowing users to control the entire ML pipeline through natural language conversations with Claude, without needing a separate client application.

## Architecture Evolution

### Previous Architecture (Iteration 5)
```
Client (main.py) <--HTTP--> Server (server.py) <---> Modules
```

### New MCP Architecture (Iteration 6)
```
Claude Desktop <--MCP/stdio--> standard_mcp_server.py <---> Modules
                                        |
                                        +---> Training (train.py)
                                        +---> Browser (browser.py)
                                        +---> CVAT API (cvat_api.py)
                                        +---> Deployment (deploy_to_cvat.py)
                                        |
                                        v
                                 Nuclio Services <--> CVAT System
```

## Key Improvements

| Feature | Iteration 5 | Iteration 6 |
|---------|-------------|-------------|
| **Architecture** | HTTP Client-Server | MCP Protocol |
| **User Interface** | Standalone Python client | Claude Desktop integration |
| **Communication** | RESTful API | stdio streams |
| **Natural Language** | API-based parsing | Native Claude understanding |
| **Installation** | Complex multi-component | Simple MCP server config |
| **User Experience** | CLI menu system | Conversational AI interface |

## Core Components

### Main Module
- **standard_mcp_server.py**: MCP server implementation handling all tool registrations and request routing

### Functional Modules (Unchanged)
- **train.py**: YOLO model training with Ultralytics
- **browser.py**: Browser automation via Playwright
- **cvat_api.py**: Complete CVAT REST API wrapper
- **deploy_to_cvat.py**: Nuclio deployment automation
- **config.py**: System configuration management

### Support Files
- **requirements_mcp.txt**: MCP-specific dependencies
- **deploy_templates/**: Nuclio function templates

## Available Tools

The MCP server exposes 11 tools to Claude:

1. **hello** - Test MCP connection
2. **train_yolo** - Train YOLOv8 models asynchronously
3. **query_task_status** - Check training task status
4. **list_tasks** - List all training tasks
5. **browser_control** - Control browser for CVAT access
6. **deploy_model** - Deploy models to Nuclio
7. **upload_data** - Upload data to CVAT
8. **auto_annotation** - Run AI-powered annotation
9. **download_dataset** - Export datasets from CVAT
10. **convert_dataset** - Convert CVAT to YOLO format
11. **download_and_convert** - One-click download and conversion
12. **view_annotations** - Open annotation interface

## Installation

### Prerequisites
- Python 3.8+
- Docker (for CVAT and Nuclio)
- Claude Desktop application
- WSL (for Windows users)

### 1. Clone Repository
```bash
git clone <repository-url>
cd YOLO_Sum/iteration-6
```

### 2. Install Dependencies
```bash
pip install -r requirements_mcp.txt
python -m playwright install
```

### 3. Setup CVAT (if not already installed)
```bash
git clone https://github.com/opencv/cvat.git
cd cvat
docker-compose up -d
```

### 4. Configure MCP in Claude Desktop

Add to Claude Desktop configuration file:
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

### 5. Configure CVAT Connection
Edit `config.py`:
```python
CVAT_URL = "http://localhost:8080"
CVAT_USERNAME = "admin"
CVAT_PASSWORD = "your_password"
```

### 6. Restart Claude Desktop
After configuration, restart Claude Desktop to load the MCP server.

## Usage Examples

### Natural Language Commands

Simply chat with Claude in natural language:

**Training:**
- "Train a YOLOv8n model for 10 epochs"
- "Use the latest converted dataset for training"
- "Check the status of my training task"

**Data Management:**
- "Upload images from C:\my_data to CVAT"
- "Download dataset from task 14 in COCO format"
- "Convert the downloaded dataset to YOLO format"

**Model Deployment:**
- "Deploy the latest trained model to CVAT"
- "Use my custom model at /path/to/model.pt"

**Annotation:**
- "Run auto-annotation on task 15"
- "Open the annotation interface for task 20"

### Workflow Examples

#### Complete Training Pipeline
1. "Upload my images to CVAT as a new task"
2. "Run auto-annotation on the task using the default model"
3. "Open the annotation interface to review results"
4. "Download and convert the dataset to YOLO format"
5. "Train a YOLOv8n model with the converted dataset"
6. "Deploy the trained model back to CVAT"

#### Quick Annotation
1. "Create a new task and upload data from my folder"
2. "Auto-annotate with the latest deployed model"
3. "Show me the annotation results"

## Technical Implementation

### MCP Protocol
- Uses standard MCP SDK for Python
- Communication via stdio streams
- Asynchronous tool execution with threading
- Real-time status updates

### Task Management
- Persistent task tracking in `tasks.json`
- Asynchronous training with progress monitoring
- Comprehensive logging in `mcp_server.log`

### Browser Automation
- Playwright-based browser control
- Automatic CVAT login
- Support for Chrome, Edge, Firefox

## File Structure
```
/iteration-6
  ├── standard_mcp_server.py  # MCP server
  ├── train.py               # Training module
  ├── browser.py             # Browser automation
  ├── cvat_api.py           # CVAT API wrapper
  ├── deploy_to_cvat.py     # Deployment module
  ├── config.py             # Configuration
  ├── requirements_mcp.txt   # Dependencies
  ├── deploy_templates/      # Nuclio templates
  │   ├── function.yaml
  │   └── main.py
  ├── tasks.json            # Task tracking (generated)
  ├── runs/                 # Training outputs (generated)
  ├── downloads/            # Downloaded datasets (generated)
  └── datasets/             # Converted datasets (generated)
```

## Troubleshooting

### MCP Connection Issues
1. Check Claude Desktop logs for MCP errors
2. Verify Python path in configuration
3. Ensure all dependencies are installed
4. Check `mcp_server.log` for detailed errors

### CVAT Connection Failed
1. Verify Docker containers are running: `docker ps`
2. Check CVAT is accessible at http://localhost:8080
3. Confirm credentials in `config.py`

### Training Failures
1. Check GPU availability for CUDA acceleration
2. Verify dataset paths and formats
3. Review training logs in `runs/logs/`

### Browser Automation Issues
1. Update Playwright browsers: `python -m playwright install`
2. Try different browser types if one fails
3. Check for conflicting browser instances

## Performance Tips

- Use GPU acceleration for training and inference
- Adjust batch sizes based on available memory
- Enable model caching for repeated deployments
- Use appropriate worker counts for data processing

## Migration from Iteration 5

1. Remove old HTTP client (`main.py`) and server (`server.py`)
2. Install MCP dependencies from `requirements_mcp.txt`
3. Configure Claude Desktop with MCP server path
4. All existing modules remain compatible

## Contributing

Contributions welcome! Areas for improvement:
- Additional model architectures support
- Extended annotation types
- Performance optimizations
- Enhanced error handling

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Anthropic MCP SDK](https://github.com/anthropics/mcp)
- Powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Integrated with [CVAT](https://github.com/opencv/cvat)
- Browser automation by [Playwright](https://playwright.dev)
