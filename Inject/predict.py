import cv2
from ultralytics import YOLO

def predict_single_image():
    # 1. 加载成功运行的那个模型权重
    model_path = r'finetune\yolov8m_final.pt' 
    model = YOLO(model_path)
    
    # 2. 读取你想测试的单张图片 (以测试集里的 大维c_1.jpg 为例)
    img_path = r'Dataset\test\可可碱_1.jpg' 
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"找不到图片: {img_path}")
        return

    # 3. 运行推理
    # classes=[1] 参数的作用是：告诉YOLO只输出属于'abnormal'（类别索引为1）的预测结果
    results = model.predict(source=img_path, classes=[1], conf=0.25)
    
    # 4. 取出第一张图片的结果并自定义绘制
    result = results[0]
    for box in result.boxes:
        # 获取框的左上角和右下角坐标，转换为整数
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # [自定义画框以避免遮挡]
        # thickness=1 代表极细线框，(0, 0, 255) 为红色边框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # [自定义写字为 "1"]
        # fontScale=0.5 和 thickness=1 保证字体很小不抢视线，放置在框左上角的偏上方(y1-5)
        cv2.putText(img, 'abnormal', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # 5. 保存并输出
    output_path = 'custom_predict_output.jpg'
    cv2.imwrite(output_path, img)
    print(f"预测完成！自定义结果已保存至项目根目录的: {output_path}")

if __name__ == "__main__":
    predict_single_image()