import os
import cv2
from ultralytics import YOLO

def predict_all_images():
    # 1. 加载成功运行的那个模型权重
    model_path = r'finetune\yolov8m_final.pt' 
    model = YOLO(model_path)
    
    # 2. 定义源图片根目录和输出目录
    dataset_base = 'Dataset'
    output_dir = 'predict_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历 Dataset 下的 train 和 test 文件夹
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_base, split)
        
        if not os.path.exists(split_dir):
            continue
            
        print(f"\n正在处理 {split} 目录: {split_dir}")
        
        # 遍历目录下的所有图片
        for filename in os.listdir(split_dir):
            if not filename.endswith('.jpg'):
                continue
                
            img_path = os.path.join(split_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  [警告] 无法读取图片: {img_path}")
                continue
            
            print(f"  预测图片: {filename}")
            
            # 3. 运行推理，仅预测类别 1 (abnormal)
            results = model.predict(source=img_path, classes=[1], conf=0.25, verbose=False)
            
            # 4. 取出第一张图片的结果并自定义绘制
            result = results[0]
            for box in result.boxes:
                # 获取框的左上角和右下角坐标，转换为整数
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 画极细的红框 (thickness=1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # 写极小的数字 "1"
                cv2.putText(img, '1', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 5. 保存结果到带前缀的文件中 (为了区分train和test同名文件)
            output_filename = f"{split}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, img)

    print(f"\n全部预测完成！结果已保存在: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    predict_all_images()