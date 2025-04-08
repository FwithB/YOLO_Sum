from ultralytics import YOLO
import multiprocessing
import os

# 设置为UTF-8编码
os.environ["PYTHONIOENCODING"] = "utf-8"

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 添加这一行解决错误
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 开始训练
    results = model.train(
        data='coco128.yaml',  # 使用内置配置
        epochs=1,
        imgsz=640,
        name='yolov8n_custom'
    )