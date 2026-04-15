import os
import json
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import shutil

"""
watch -n 3 nvidia-smi

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=4 python main.py

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=4 nohup python main.py > /dev/null 2>&1 &

"""

# 1. 标签格式转换
def convert_labelme_to_yolo(json_path, img_path, output_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return
    h, w = img.shape[:2]
    
    with open(output_txt_path, 'w') as f:
        for shape in data['shapes']:
            label = int(shape['label'])
            label = max(0, min(label, 3))
            
            # 映射标签到正确的YOLO格式
            # YOLO要求索引从0开始，0对应'normal'，1对应'abnormal'
            # 假设原始标签：0-normal，1,2,3-abnormal
            if label >= 1:  # 如果是1,2,3，都映射为abnormal类
                yolo_label = 1  # abnormal类
            else:  # 如果是0，映射为normal类
                yolo_label = 0  # normal类
                
            points = shape['points']
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])
            
            # 确保坐标在合理范围内
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Invalid bbox in {json_path}, skipping")
                continue
                
            # 计算中心点和宽高
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # 确保归一化坐标在(0,1)范围内
            x_center = max(0.0, min(x_center, 1.0))
            y_center = max(0.0, min(y_center, 1.0))
            width = max(0.001, min(width, 1.0))
            height = max(0.001, min(height, 1.0))
            
            f.write(f"{yolo_label} {x_center} {y_center} {width} {height}\n")

def check_labels(label_dir):
    """检查标签文件格式是否正确"""
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            txt_path = os.path.join(label_dir, file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Error in {txt_path} line {i}: {line}")
                        continue
                    
                    label = int(parts[0])
                    if label < 0 or label >= 2:  # 应该只有0或1
                        print(f"Error in {txt_path}: label {label} out of range (0-1)")
                    
                    # 检查坐标
                    for j in range(1, 5):
                        val = float(parts[j])
                        if j in [1, 2]:  # x_center, y_center
                            if val < 0 or val > 1:
                                print(f"Warning in {txt_path}: coordinate {val} out of [0,1] range")
                        else:  # width, height
                            if val <= 0 or val > 1:
                                print(f"Warning in {txt_path}: size {val} out of (0,1] range")

# 转换所有标签
def convert_all_labels(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            base = file[:-5]  # remove .json
            json_path = os.path.join(data_dir, file)
            img_path = os.path.join(data_dir, base + '.jpg')
            txt_path = os.path.join(output_dir, base + '.txt')
            img_out_path = os.path.join(output_dir, base + '.jpg')
            if os.path.exists(img_path):
                convert_labelme_to_yolo(json_path, img_path, txt_path)
                shutil.copy(img_path, img_out_path)  # 复制图片


def find_checkpoint(checkpoint_root: str) -> str:
    candidates = []
    for root, _, files in os.walk(checkpoint_root):
        for file in files:
            if file.endswith('.pt'):
                candidates.append(os.path.join(root, file))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint file found under {checkpoint_root}')

    for name in ['best.pt', 'last.pt', 'yolov8m.pt', 'weights.pt']:
        for candidate in candidates:
            if os.path.basename(candidate) == name:
                return candidate

    return sorted(candidates)[-1]

# 3. 训练
def train_model(data_yaml, epochs=1000, batch_size=16):
    model = YOLO(r'C:\\Users\\ASUS\\Desktop\\Inject\\checkpoints\\yolov8m_final.pt')  
    
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.001,  # 降低最终学习率
        momentum=0.937,
        weight_decay=0.0005,  # 增加权重衰减
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # box loss增益
        cls=0.5,  # cls loss增益
        dfl=1.5,  # dfl loss增益
        save_period=5,
        project='checkpoints',
        name='yolov8m',
        patience=50,  # 早停耐心
        dropout=0.1,  # 添加dropout防止过拟合
        hsv_h=0.015,  # 颜色增强
        hsv_s=0.7,    # 饱和度增强
        hsv_v=0.4,    # 明度增强
        degrees=0.1,  # 旋转增强
        translate=0.1, # 平移增强
        scale=0.5,      # 缩放增强
        shear=0.1,      # 剪切增强
        perspective=0.0, # 透视增强
        flipud=0.0,     # 上下翻转增强
        fliplr=0.5,     # 左右翻转增强

        # 对异常类进行重点增强的参数
        mixup=0.2,           # MixUp增强，有助于处理类别不平衡
        copy_paste=0.3,      # Copy-Paste增强，特别对小目标有效
    )

    final_path = os.path.join('checkpoints', 'yolov8m_final.pt')
    try:
        model.save(final_path)
        return final_path
    except TypeError:
        # 如果 save 接口无法使用，则回退到训练生成的 checkpoint
        fallback = find_checkpoint(os.path.join('checkpoints', 'yolov8m'))
        print(f"Warning: model.save failed, using fallback checkpoint: {fallback}")
        return fallback

# 4. 日志记录
def log_metrics(model, epoch, train_loss, val_loss, log_dir):
    # 二分类指标（现在模型是二分类）
    results = model.val()
    mAP50 = results.box.map50
    mAP5095 = results.box.map
    precision = results.box.mp
    recall = results.box.mr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
        f.write(f"{epoch},{train_loss},{val_loss}\n")
    
    with open(os.path.join(log_dir, 'tscore.txt'), 'a') as f:
        f.write(f"{epoch},{mAP50},{mAP5095},{precision},{recall},{f1}\n")
    
    # 二分类指标（直接使用，因为现在是二分类）
    preds = []
    targets = []
    for result in results:
        for pred in result.boxes:
            cls = int(pred.cls)
            preds.append(cls)
        for target in result.boxes:
            cls = int(target.cls)
            targets.append(cls)
    
    if preds and targets:
        precision_2c, recall_2c, f1_2c, _ = precision_recall_fscore_support(targets, preds, average='binary')
        ap50 = average_precision_score(targets, preds)  # 近似mAP@0.5
        ap5095 = ap50  # 简化
        
        with open(os.path.join(log_dir, '2c_score.txt'), 'a') as f:
            f.write(f"{epoch},{ap50},{ap5095},{precision_2c},{recall_2c},{f1_2c}\n")

# 5. 评估
def evaluate_model(model_path, data_yaml, results_dir):
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    mAP50 = results.box.map50
    mAP5095 = results.box.map
    precision = results.box.mp
    recall = results.box.mr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    with open(os.path.join(results_dir, 'tscore.txt'), 'w') as f:
        f.write(f"{mAP50},{mAP5095},{precision},{recall},{f1}\n")
    
    # 二分类（现在是二分类，直接使用）
    metrics = getattr(results, 'results_dict', {})
    precision_2c = metrics.get('metrics/precision(B)')
    recall_2c = metrics.get('metrics/recall(B)')
    ap50_2c = metrics.get('metrics/mAP50(B)')
    ap5095_2c = metrics.get('metrics/mAP50-95(B)')
    
    if precision_2c is not None and recall_2c is not None:
        f1_2c = 2 * precision_2c * recall_2c / (precision_2c + recall_2c) if (precision_2c + recall_2c) > 0 else 0
        ap5095_2c = ap5095_2c if ap5095_2c is not None else ap50_2c
        with open(os.path.join(results_dir, '2c_score.txt'), 'w') as f:
            f.write(f"{ap50_2c},{ap5095_2c},{precision_2c},{recall_2c},{f1_2c}\n")
    else:
        with open(os.path.join(results_dir, '2c_score.txt'), 'w') as f:
            f.write("0,0,0,0,0\n")

# 主函数
if __name__ == "__main__":
    
    # 重新生成 data.yaml
    data_yaml = f"""
        train: {os.path.abspath('labels/train')}
        val: {os.path.abspath('labels/test')}
        test: {os.path.abspath('labels/test')}

        nc: 2
        names: ['normal', 'abnormal']
        """
    
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)
    
    # 【注释掉训练】
    # checkpoint_path = train_model('data.yaml')

    print(f"\n\n开始评估...\n\n")
    
    # 【直接指定你实际训练好的有效模型路径进行评估】
    # 请确认这个路径下的模型是完好的
    checkpoint_path = r'finetune\yolov8m_final.pt' 
    
    # 评估
    evaluate_model(checkpoint_path, 'data.yaml', 'results')
