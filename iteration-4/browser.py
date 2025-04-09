"""
浏览器控制模块 - 使用Playwright实现浏览器自动化
"""

import asyncio
import logging
from playwright.async_api import async_playwright
import os
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('browser_controller.log')
    ]
)
logger = logging.getLogger('browser_controller')

class BrowserController:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_initialized = False
    
    async def initialize(self, browser_type='msedge', headless=False, force_new=False):
        """初始化浏览器"""
        # 如果要求强制创建新实例，或检测到浏览器状态异常，则关闭现有实例
        if force_new and self.is_initialized:
            logger.info("强制创建新浏览器实例，关闭现有实例")
            await self.close()
        
        # 尝试验证现有浏览器状态
        if self.is_initialized:
            try:
                # 尝试执行简单操作，验证浏览器是否真的可用
                logger.info("检查浏览器实例是否可用...")
                current_url = self.page.url  # 这会在浏览器不可用时抛出异常
                logger.info(f"浏览器实例状态正常，当前URL: {current_url}")
                return {"status": "success", "message": "浏览器已经初始化且状态正常"}
            except Exception as e:
                logger.warning(f"浏览器状态异常，将重新初始化: {str(e)}")
                # 浏览器状态异常，重置状态并重新初始化
                await self.close()
        
        try:
            logger.info(f"开始初始化 {browser_type} 浏览器, headless={headless}")
            self.playwright = await async_playwright().start()
            
            # 添加浏览器启动前日志
            logger.info("正在启动浏览器...")
            
            # 配置浏览器启动参数，提高稳定性
            browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
            
            # 根据浏览器类型启动不同浏览器
            if browser_type.lower() == 'chromium':
                browser_instance = self.playwright.chromium
                self.browser = await browser_instance.launch(
                    headless=headless,
                    args=browser_args
                )
            elif browser_type.lower() == 'edge' or browser_type.lower() == 'msedge':
                browser_instance = self.playwright.chromium
                self.browser = await browser_instance.launch(
                    headless=headless,
                    channel="msedge",
                    args=browser_args
                )
            elif browser_type.lower() == 'firefox':
                browser_instance = self.playwright.firefox
                self.browser = await browser_instance.launch(headless=headless)
            elif browser_type.lower() == 'webkit':
                browser_instance = self.playwright.webkit
                self.browser = await browser_instance.launch(headless=headless)
            else:
                logger.error(f"不支持的浏览器类型: {browser_type}")
                return {"status": "error", "message": f"不支持的浏览器类型: {browser_type}"}
            
            # 添加浏览器启动后日志
            logger.info(f"{browser_type} 浏览器启动完成")
            
            # 设置浏览器上下文
            logger.info("创建浏览器上下文...")
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800}
            )
            
            # 创建新页面
            logger.info("创建新页面...")
            self.page = await self.context.new_page()
            
            # 设置页面事件监听，帮助调试
            self.page.on("console", lambda msg: logger.info(f"浏览器控制台: {msg.text}"))
            self.page.on("pageerror", lambda err: logger.error(f"页面错误: {err}"))
            
            # 验证页面可正常访问
            await self.page.goto("about:blank")
            logger.info("页面创建成功并可正常访问")
            
            self.is_initialized = True
            
            # 添加一个可见的操作来确认浏览器窗口确实打开
            if not headless:
                # 打开一个明显的页面，确认可以看到
                await self.page.goto("data:text/html,<html><body><h1 style='color:red; font-size:50px;'>浏览器控制系统已启动</h1></body></html>")
                logger.info("浏览器显示测试页面")
                await asyncio.sleep(1)  # 短暂停留，让用户看到
            
            return {"status": "success", "message": f"浏览器初始化成功: {browser_type}"}
        
        except Exception as e:
            logger.error(f"浏览器初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 确保所有资源被释放
            try:
                if self.page:
                    await self.page.close()
                if self.context:
                    await self.context.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()
            except Exception as close_err:
                logger.error(f"清理资源时出错: {str(close_err)}")
            
            # 重置状态
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
            self.is_initialized = False
            
            return {"status": "error", "message": f"浏览器初始化失败: {str(e)}"}
    
    async def navigate(self, url):
        """导航到指定URL"""
        if not self.is_initialized:
            logger.info("Browser not initialized, initializing now")
            init_result = await self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        try:
            await self.page.goto(url)
            current_url = self.page.url
            logger.info(f"Navigated to: {current_url}")
            return {"status": "success", "message": f"已导航至: {current_url}"}
        
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            return {"status": "error", "message": f"导航到 {url} 失败: {str(e)}"}


    async def login_cvat(self, username="admin", password="Yyh277132984"):
        """登录CVAT - 适配两步登录流程，使用Chrome浏览器"""
        if not self.is_initialized:
            logger.info("浏览器未初始化，现在初始化")
            init_result = await self.initialize(browser_type="chromium")  # 使用Chrome
            if init_result["status"] == "error":
                return init_result
        
        try:
            # 确保导航到登录页面
            await self.navigate("http://localhost:8080/auth/login")
            
            # 等待页面完全加载
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            # 第一步：清空并输入用户名
            logger.info("第一步：输入用户名")
            username_field = await self.page.query_selector('input[placeholder="Email or username"]')
            if not username_field:
                username_field = await self.page.query_selector('input[type="text"]')
            
            if username_field:
                # 先清空字段
                await username_field.click()
                await username_field.press('Control+a')
                await username_field.press('Delete')
                
                # 输入用户名
                await username_field.type(username)
                logger.info(f"已输入用户名: {username}")
                
                # 按Enter提交用户名
                await username_field.press('Enter')
                logger.info("已按Enter键提交用户名")
                
                # 等待密码输入框出现
                await asyncio.sleep(2)
                
                # 第二步：输入密码
                logger.info("第二步：寻找并输入密码框")
                password_field = await self.page.query_selector('input[type="password"]')
                
                if password_field:
                    logger.info("找到密码输入框")
                    # 确保密码框获得焦点
                    await password_field.click()
                    
                    # 清空可能存在的内容
                    await password_field.press('Control+a')
                    await password_field.press('Delete')
                    
                    # 输入密码
                    await password_field.type(password)
                    logger.info("已输入密码")
                    
                    # 点击"Next"按钮
                    next_button = await self.page.query_selector('button:has-text("Next")')
                    if next_button:
                        await next_button.click()
                        logger.info("已点击Next按钮")
                    else:
                        await password_field.press('Enter')
                    
                    # 等待登录完成
                    await asyncio.sleep(5)
                    
                    # 检查是否登录成功
                    is_logged_in = await self.check_login_success()
                    if is_logged_in:
                        return {"status": "success", "message": "成功登录CVAT"}
                    else:
                        return {"status": "warning", "message": "登录操作已执行，但可能未成功，请检查浏览器"}
                else:
                    logger.error("未找到密码输入框")
                    return {"status": "error", "message": "未找到密码输入框"}
            else:
                logger.error("未找到用户名输入框")
                return {"status": "error", "message": "未找到用户名输入框"}
        
        except Exception as e:
            logger.error(f"登录CVAT过程中出错: {str(e)}")
            return {"status": "error", "message": f"登录CVAT失败: {str(e)}"}

    async def check_login_success(self):
        """检查是否成功登录CVAT"""
        try:
            logged_in_element = await self.page.query_selector('.cvat-header-menu-user-dropdown')
            return logged_in_element is not None
        except Exception:
            return False

    async def create_project(self, name="新项目", labels=None):
        """创建新项目"""
        if not self.is_initialized:
            init_result = await self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        if labels is None:
            labels = [{"name": "person"}, {"name": "car"}]
        
        try:
            # 导航到项目页面
            await self.navigate("http://localhost:8080/projects")
            
            # 点击创建项目按钮
            await self.page.click('.cvat-create-project-button')
            
            # 填写项目名称
            await self.page.fill('input#name', name)
            
            # 添加标签
            for label in labels:
                await self.page.click('.cvat-constructor-viewer-new-item')
                await self.page.fill('.cvat-label-constructor-creator input', label["name"])
                await self.page.click('.cvat-label-constructor-creator-submit-button')
            
            # 提交创建
            await self.page.click('.cvat-submit-project-button')
            
            # 等待创建成功
            await self.page.wait_for_selector('.cvat-project-page', timeout=10000)
            logger.info(f"Successfully created project: {name}")
            return {"status": "success", "message": f"成功创建项目: {name}"}
        
        except Exception as e:
            logger.error(f"Failed to create project: {str(e)}")
            return {"status": "error", "message": f"创建项目失败: {str(e)}"}
    
    async def close(self):
        """关闭浏览器"""
        if not self.is_initialized:
            logger.info("No browser to close")
            return {"status": "success", "message": "没有需要关闭的浏览器"}
        
        try:
            await self.context.close()
            await self.browser.close()
            await self.playwright.stop()
            
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
            self.is_initialized = False
            
            logger.info("Browser closed")
            return {"status": "success", "message": "浏览器已关闭"}
        
        except Exception as e:
            logger.error(f"Failed to close browser: {str(e)}")
            return {"status": "error", "message": f"关闭浏览器失败: {str(e)}"}
    
    async def take_screenshot(self, path="screenshot.png"):
        """截取屏幕截图"""
        if not self.is_initialized:
            init_result = await self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        try:
            await self.page.screenshot(path=path)
            logger.info(f"Screenshot saved to: {path}")
            return {"status": "success", "message": f"截图已保存至: {path}"}
        
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return {"status": "error", "message": f"截图失败: {str(e)}"}

# 创建全局浏览器控制器实例
browser_controller = BrowserController()

# 异步函数用于处理浏览器请求
async def handle_browser_request(request_data):
    """处理来自客户端的浏览器控制请求"""
    try:
        action = request_data.get("action")
        
        if action == "initialize":
            browser_type = request_data.get("browser_type", "chromium")
            headless = request_data.get("headless", False)
            force_new = request_data.get("force_new", False)  # 添加force_new参数
            return await browser_controller.initialize(browser_type, headless)
        
        elif action == "navigate":
            url = request_data.get("url")
            if not url:
                logger.error("URL is required for navigate action")
                return {"status": "error", "message": "导航操作需要URL参数"}
            return await browser_controller.navigate(url)
        
        elif action == "login_cvat":
            username = request_data.get("username", "admin")
            password = request_data.get("password", "Yyh277132984")
            return await browser_controller.login_cvat(username, password)
        
        elif action == "create_project":
            name = request_data.get("name", "新项目")
            labels = request_data.get("labels", None)
            return await browser_controller.create_project(name, labels)
        
        elif action == "close":
            return await browser_controller.close()
        
        elif action == "take_screenshot":
            path = request_data.get("path", "screenshot.png")
            return await browser_controller.take_screenshot(path)
        
        elif action == "open_cvat":
            # 特殊操作：打开并登录CVAT
            init_result = await browser_controller.initialize(browser_type="chromium")
            if init_result["status"] == "error":
                return init_result
            
            nav_result = await browser_controller.navigate("http://localhost:8080")
            if nav_result["status"] == "error":
                return nav_result
            
            # 尝试登录
            try:
                login_result = await browser_controller.login_cvat(username="admin", password="Yyh277132984")
                return login_result
            except Exception as e:
                # 如果登录失败，至少我们打开了CVAT
                logger.warning(f"Opened CVAT but login might have failed: {str(e)}")
                return {"status": "partial", "message": "已打开CVAT但登录可能失败，请手动登录"}
        
        else:
            logger.error(f"Unknown action: {action}")
            return {"status": "error", "message": f"未知操作: {action}"}
    
    except Exception as e:
        logger.error(f"Error handling browser request: {str(e)}")
        return {"status": "error", "message": f"处理浏览器请求时出错: {str(e)}"}

# 同步包装函数，用于在非异步环境中调用
def process_browser_request(request_data):
    """处理浏览器请求的同步包装函数"""
    try:
        # 获取当前线程ID用于调试
        import threading
        thread_id = threading.current_thread().ident
        logger.info(f"处理浏览器请求 [线程ID: {thread_id}]")
        
        # 记录请求内容
        logger.info(f"请求数据: {json.dumps(request_data, ensure_ascii=False)}")
        
        # 尝试获取当前事件循环
        try:
            loop = asyncio.get_event_loop()
            logger.info("成功获取现有事件循环")
        except RuntimeError:
            # 如果当前线程没有事件循环
            logger.info("创建新事件循环")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 运行异步函数
        logger.info("开始执行异步请求...")
        result = loop.run_until_complete(handle_browser_request(request_data))
        logger.info(f"异步请求执行完成: {json.dumps(result, ensure_ascii=False)}")
        return result
    
    except Exception as e:
        logger.error(f"处理浏览器请求时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"处理浏览器请求时出错: {str(e)}"}

# 简化函数：打开CVAT
def open_cvat():
    """打开CVAT的简化函数"""
    request_data = {"action": "open_cvat"}
    return process_browser_request(request_data)

# 测试代码
if __name__ == "__main__":
    # 测试浏览器打开CVAT
    result = open_cvat()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 等待用户查看
    input("按Enter关闭浏览器...")
    
    # 关闭浏览器
    close_result = process_browser_request({"action": "close"})
    print(json.dumps(close_result, indent=2, ensure_ascii=False))
