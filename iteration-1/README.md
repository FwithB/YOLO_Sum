
---

# Python 文件搜索助手

这是一个基于 Flask 的文件搜索助手 Demo 项目，通过调用 OpenRouter API 使用 LLM 分析用户指令，自动提取目标文件扩展名，并在本地目录中搜索符合条件的文件。项目包含后端 Flask 服务和前端 HTML 页面，用于展示搜索结果和预览部分文件内容。

---

## 项目背景

在日常开发中，我们经常需要快速定位指定类型的文件。该项目通过自然语言交互方式，将用户输入的搜索指令解析为文件扩展名列表，然后在指定目录中搜索（例如 Python、PDF、Word 等文件）。同时，该 Demo 记录了调用 OpenRouter API 的过程以及搜索日志，便于调试和后续功能扩展。

---

## 项目功能

- **自然语言解析：**  
  使用 OpenRouter API（调用 LLM 模型）解析用户输入的搜索指令，返回文件扩展名列表。

- **本地文件搜索：**  
  根据解析出的扩展名，在指定目录下递归搜索符合条件的文件，并返回文件名、路径、大小、修改时间以及（对 Python 文件）预览内容。

- **前后端交互：**  
  后端使用 Flask 提供 API 接口，前端页面通过 Ajax 请求调用后端接口，实时展示搜索结果。

- **错误处理：**  
  对 API 调用超时、目录不存在、解析错误等情况做了详细日志记录，并返回错误提示。

---

## 技术栈

- **后端：** Python、Flask、Flask-CORS、OpenAI SDK（用于 OpenRouter API 调用）
- **前端：** HTML、CSS、JavaScript（使用 Fetch API 进行异步请求）

---

## 项目结构

```
PYtracking/
├── app.py         # Flask 服务代码，包含文件搜索与 API 调用逻辑
├── index.html     # 前端展示页面，提供搜索输入框与结果展示
├── requirements.txt  # 项目依赖列表
└── README.md      # 项目说明文档
```

- **app.py**  
  启动 Flask 服务器，提供 `/search` POST 接口。接口接收 JSON 格式的参数（包含搜索指令和目录路径），调用 LLM 模型解析指令，并在本地目录中搜索文件。

- **index.html**  
  简单的前端页面，包含输入搜索指令和目录的控件，以及展示搜索结果的区域。通过 JavaScript 调用后端接口并展示返回数据。

- **requirements.txt**  
  列出所需依赖项，例如：  
  ```txt
  Flask
  flask-cors
  openai
  ```

---

## 安装与配置

1. **克隆项目到本地**  
   将仓库克隆到你的本地目录：
   ```bash
   git clone https://github.com/FwithB/PYtracking.git
   cd PYtracking
   ```

2. **安装依赖**  
   确保你已经安装了 Python 3，然后执行：
   ```bash
   pip install -r requirements.txt
   ```

3. **配置 OpenRouter API 密钥**  
   在 `app.py` 中找到以下代码：
   ```python
   API_KEY = ""
   ```
   将 `API_KEY` 替换为你的 OpenRouter API 密钥。

4. **（可选）调整搜索目录**  
   默认搜索目录在接口参数中传入（例如：`/path/to/default/directory`），你可以在前端页面的输入框中指定你要搜索的目录。

---

## 运行项目

1. **启动 Flask 服务**  
   在项目根目录下运行：
   ```bash
   python app.py
   ```
   Flask 服务会启动在 `http://localhost:5000`，并开启调试模式（debug=True）。

2. **打开前端页面**  
   使用浏览器打开 `index.html` 文件（例如，直接双击文件或通过本地 Web 服务器访问）。在页面上输入搜索目录和搜索指令，然后点击“搜索”按钮即可开始搜索。

3. **接口调用说明**  
   前端页面会通过 POST 请求调用接口：
   ```
   POST http://localhost:5000/search
   ```
   请求体格式为 JSON，例如：
   ```json
   {
       "instruction": "找出所有Python文件",
       "directory": "C:/Users/Documents"
   }
   ```
   接口返回的 JSON 数据包含搜索结果列表（文件名、路径、大小、修改时间、预览信息等）。

---

## 使用示例

在前端页面中：
- **目录输入框：** 输入你想搜索的目录路径（如 `C:/Users/Documents`）。
- **搜索指令输入框：** 输入搜索指令，例如 “找出所有 Python 文件”。
- **搜索结果展示：** 页面会展示搜索到的文件列表及相关信息，对于 Python 文件还会额外展示前 5 行预览内容。

---

## 未来规划

- **功能扩展：**
  - 支持更多自然语言解析能力，增加文件类型的自动映射和扩展。
  - 增加多目录搜索和分页显示结果的功能。
- **前端优化：**
  - 提供更丰富的交互体验，例如搜索进度条、错误提示弹窗等。
- **日志与调试：**
  - 将日志记录完善化，支持日志文件存储，方便后续排查问题。

---

## 贡献与反馈

如果你有任何建议或发现问题，欢迎通过 [GitHub Issues](https://github.com/FwithB/PYtracking/issues) 提出改进意见。非常感谢你的关注与支持！

---

## 许可证

本项目采用 MIT 许可证，详情请查看 [LICENSE](LICENSE) 文件。

---
