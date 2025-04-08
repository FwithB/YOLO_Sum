
---

# YOLO Agent 训练助手

这是一个基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的 AI Agent 项目，旨在通过自然语言指令控制 YOLO 模型的训练流程。项目集成了 OpenRouter API（调用 OpenAI GPT-3.5 Turbo 模型）用于将用户输入的自然语言解析为相应的函数调用，从而自动启动模型训练并实时显示训练日志。对于调试和错误记录，该项目也详细记录了训练过程中的输出，便于排查问题。

![Screenshot 2025-04-02 161138](https://github.com/user-attachments/assets/804a4b4e-7902-4645-8a49-322a1586f296)
![Screenshot 2025-04-02 161158](https://github.com/user-attachments/assets/4e9ef437-d37d-4773-8ec5-b504ca0f96bd)

---

## 项目背景与动机

- **项目背景：**  
  在模型训练和应用过程中，通过命令行交互以自然语言指令启动和管理训练流程，可以大大降低操作门槛，提高项目的易用性。  
- **动机：**  
  1. 学习并实践如何将自然语言处理与实际模型训练相结合；  
  2. 形成一个可扩展、模块化的 AI Agent 项目，为未来的开发和简历项目积累经验；  
  3. 记录训练、调试和错误日志，方便后期维护与改进。

---

## 项目结构

```
YOLO-Agent/
├── main.py           # 项目的入口文件，处理用户交互和指令解析
├── train.py          # 训练模块，负责调用 Ultralytics YOLO 进行训练
├── requirements.txt  # 项目依赖列表（例如：ultralytics, openai 等）
├── README.md         # 项目说明文档
└── .gitignore        # Git忽略配置（过滤不必要上传的文件，如缓存、权重等）
```

- **main.py：**  
  - 通过命令行等待用户输入自然语言指令。
  - 利用 OpenRouter API 将指令转换为函数调用（例如：`train_yolo`）。
  - 执行对应函数，并实时输出训练日志。
- **train.py：**  
  - 使用 `ultralytics` 提供的 YOLO API 加载预训练模型并进行训练。
  - 示例中使用的是 `yolov8n.pt` 预训练权重和内置的 `coco128.yaml` 数据集配置，训练 1 个 epoch 作为演示。
- **.gitignore：**  
  - 过滤掉生成的缓存文件、训练输出（如 `runs/` 目录、`.pt` 文件）以及其他不必要上传的内容。

---

## 环境与依赖

- **Python 版本：** 3.11 或更高
- **主要依赖：**
  - [ultralytics](https://pypi.org/project/ultralytics/): YOLOv8 模型训练与推理库
  - [openai](https://pypi.org/project/openai/): 连接 OpenRouter API（需替换为你使用的API客户端）
  - 其他常用库：`json`、`subprocess`、`multiprocessing`、`os` 等

在项目根目录下创建一个 `requirements.txt` 文件，示例如下：

```txt
ultralytics
openai
```

安装依赖命令：

```bash
pip install -r requirements.txt
```

---

## 使用方法

1. **克隆项目并安装依赖：**

   ```bash
   git clone https://github.com/你的用户名/YOLO-Agent.git
   cd YOLO-Agent
   pip install -r requirements.txt
   ```

2. **配置 API 密钥：**  
   在 `main.py` 文件中，替换 `API_KEY` 为你自己的 OpenRouter API 密钥。

3. **启动训练助手：**

   ```bash
   python main.py
   ```

   程序启动后，你会看到类似以下的信息：

   ```
   [SUCCESS] YOLO训练助手已启动 - 请用自然语言输入指令
   [INFO] 例如: "帮我训练一个yolov8模型" 或 "开始模型训练"
   [INFO] 输入'退出'结束程序
   --------------------------------------------------
   请输入指令:
   ```

4. **输入自然语言指令：**  
   例如输入：
   
   ```
   帮我训练一个yolov8模型
   ```

   程序会调用 OpenRouter API，将自然语言解析为函数调用（例如 `train_yolo`），并启动训练进程。你可以在控制台实时看到训练日志和错误信息。

---

## 代码详解

### main.py

- **指令处理：**  
  - `process_instruction` 函数调用 OpenRouter API，将用户指令转换为函数调用。
  - 如果解析成功，则根据返回的函数名和参数执行对应的函数；否则，默认启动训练。

- **实时日志输出：**  
  - 通过 `subprocess.Popen` 启动训练进程，并实时读取标准输出与错误，便于追踪训练进度和调试。

### train.py

- **训练流程：**  
  - 使用 `ultralytics.YOLO` 加载预训练模型 `yolov8n.pt`。
  - 调用 `model.train()` 方法进行训练，参数包括数据集配置（`coco128.yaml`）、训练 epoch 数、图像大小和训练任务名称。
  - 使用 `multiprocessing.freeze_support()` 解决 Windows 平台下的多进程问题。

---

## 错误处理与日志记录

- **日志输出：**  
  程序中定义了 `print_info`、`print_success`、`print_error`、`print_warning` 等函数，对不同级别的消息进行彩色输出，便于在控制台区分信息。

- **错误处理：**  
  - 如果训练过程中遇到错误，会输出错误码及详细错误信息，方便定位问题。
  - 如果 OpenRouter API 未返回预期的工具调用，程序会默认执行训练流程，并通过警告信息提示用户。

---

## 未来计划

- **功能扩展：**
  - 增加更多自然语言指令支持，扩展更多训练、推理和监控功能。
  - 集成更多数据集和模型版本，支持自定义参数配置。
- **项目优化：**
  - 改进日志管理机制，支持日志文件存储与分析。
  - 增加 Web 界面，提供更加友好的用户交互体验。
- **项目维护：**
  - 持续更新 README 与文档，记录项目发展和变更历史。
  - 开放 Issue 与 Pull Request，鼓励社区和其他开发者参与改进。

---

## 贡献与许可证

- **贡献：**  
  欢迎任何形式的贡献，无论是代码改进、文档完善还是问题反馈，请通过 GitHub Issue 或 Pull Request 与我联系。

- **许可证：**  
  本项目采用 MIT 许可证，详细内容请参考 [LICENSE](LICENSE) 文件。

---

