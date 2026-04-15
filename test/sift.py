import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class DefectDetector:
    def __init__(self, template_path):
        """
        初始化缺陷检测器
        """
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError("无法读取模板图像")
        
        # 保存原始模板用于大范围初步定位，避免预处理干扰
        self.raw_template = self.template.copy()
        
        # 预处理模板并输出其过程图
        self.template = self._preprocess_image(self.template, save_debug=True, debug_prefix="debug_template")
        self.h, self.w = self.template.shape
        
        # --- 生成字符专注掩码 (Character Mask) ---
        # 利用梯度找到只有文字/笔画的区域，忽略平滑的玻璃反光
        grad_x = cv2.Sobel(self.template, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.template, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # 二值化梯度图并膨胀，形成包裹住所有打印字符的白色掩盖区
        _, char_mask = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)
        kernel_mask = np.ones((7, 7), np.uint8)
        self.char_mask = cv2.dilate(char_mask, kernel_mask, iterations=2)
        
        # 初始化SIFT（使用更稳定的参数）
        self.sift = cv2.SIFT_create(
            nfeatures=0,           # 不限制特征点数量
            nOctaveLayers=3,       # 每层金字塔的层数
            contrastThreshold=0.04, # 对比度阈值（降低以获取更多特征点）
            edgeThreshold=10,       # 边缘阈值
            sigma=1.6              # 高斯金字塔的sigma值
        )
        
        # 提取模板特征（只在字符区域内提取，提高匹配质量）
        self.kp_template, self.des_template = self.sift.detectAndCompute(
            self.template, self.char_mask
        )
        
        # 创建FLANN匹配器（使用更精确的参数）
        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5
        )
        search_params = dict(
            checks=100,            # 增加检查次数提高精度
            eps=0.01               # 精度控制
        )
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        print(f"模板特征点数量: {len(self.kp_template)}")
        print(f"字符掩码区域大小: {np.sum(self.char_mask>0)} 像素")
    
    def _preprocess_image(self, img, save_debug=False, debug_prefix="debug"):
        """
        图像预处理：增强对比度、去噪
        添加了生成并保存中间过程图片的功能
        """
        img_orig = img.copy()
        
        # 1. 使用CLAHE增强局部对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)
        
        # 2. 双边滤波去噪（保留边缘）
        img_bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        
        # 3. 锐化增强细节
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        img_sharp = cv2.filter2D(img_bilateral, -1, kernel)
        
        if save_debug:
            # 单独保存每一阶段
            cv2.imwrite(f"{debug_prefix}_00_original.jpg", img_orig)
            cv2.imwrite(f"{debug_prefix}_01_clahe.jpg", img_clahe)
            cv2.imwrite(f"{debug_prefix}_02_bilateral.jpg", img_bilateral)
            cv2.imwrite(f"{debug_prefix}_03_sharpened.jpg", img_sharp)
            
            # 将四张图加上简单的文字提示并横向拼接成一张长图，方便对比
            def add_label(image, text):
                res = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.putText(res, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return res
            
            panel = np.hstack([
                add_label(img_orig, "Original"),
                add_label(img_clahe, "CLAHE"),
                add_label(img_bilateral, "Bilateral"),
                add_label(img_sharp, "Sharpened")
            ])
            cv2.imwrite(f"{debug_prefix}_all_steps.jpg", panel)
            print(f"[*] 预处理调试图已生成: {debug_prefix}_all_steps.jpg")
            
        return img_sharp
    
    def align_image(self, test_img):
        """
        将测试图像对齐到模板
        修改：接受图像数组而不是路径，方便多瓶子检测
        """
        # 预处理测试图
        test_img_processed = self._preprocess_image(test_img, save_debug=True, debug_prefix="debug_test_roi")
        
        # 提取测试图特征
        kp_test, des_test = self.sift.detectAndCompute(test_img_processed, None)
        
        if des_test is None or len(kp_test) < 10:
            return None, None
        
        # 特征匹配
        matches = self.flann.knnMatch(self.des_template, des_test, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, None
        
        # 提取匹配点坐标
        src_pts = np.float32([self.kp_template[m.queryIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_test[m.trainIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        
        # 修改：让 findHomography 直接计算 模板 -> 测试图 (Template -> Test) 的正向矩阵
        M, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995
        )
        
        if M is None:
            return None, None
        
        # 修改：使用 WARP_INVERSE_MAP 让 OpenCV 原生且安全地完成 Test -> Template 的对齐
        aligned_test = cv2.warpPerspective(
            test_img_processed, M, (self.w, self.h),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 这时返回的 M 就是 Template -> Test 的正向矩阵
        return aligned_test, M
    
    def detect_defects_single(self, test_img):
        """
        检测单个图像的缺陷
        返回：缺陷列表和对齐后的图像
        """
        # 对齐图像
        aligned_test, M = self.align_image(test_img)
        
        if aligned_test is None:
            return [], None
        
        # 计算绝对差异
        diff = cv2.absdiff(self.template, aligned_test)
        
        # 稍微加大模糊，吸收1~2像素以内的微小对齐误差
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # 判定缺陷：差异灰度大于 50 的像素才认为是缺陷 (根据实际可调整)
        _, thresh = cv2.threshold(diff_blur, 50, 255, cv2.THRESH_BINARY)
        
        # 应用掩码，只在乎字符区域内的差异
        thresh = cv2.bitwise_and(thresh, thresh, mask=self.char_mask)
        
        # 形态学操作：去除小面积噪点，将相近的缺陷笔画连通
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 使用固定面积阈值过滤噪点，根据你的图幅，小于 10~20 的通常是噪点
            if area > 15:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        return defects, M
    
    def detect_multiple_bottles(self, test_img_path, threshold=0.45):
        """
        处理包含多个针剂瓶的测试图
        利用原始模板进行宏观定位，再分别进行裁剪与精准对齐检测
        """
        test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        test_img_color = cv2.imread(test_img_path)
        
        if test_img is None:
            raise ValueError("无法读取测试图像")
        
        # 1. 使用原始图像和标准模板匹配，粗略定位所有瓶子
        res = cv2.matchTemplate(test_img, self.raw_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        # 合并重叠的边界框 (非极大值抑制)
        rectangles = []
        for pt in zip(*loc[::-1]):
            rectangles.append([int(pt[0]), int(pt[1]), int(self.w), int(self.h)])
        bottle_boxes, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)
        
        print(f"检测到 {len(bottle_boxes)} 个瓶子")
        
        # 2. 对每个瓶子进行缺陷检测
        result_img = test_img_color.copy()
        total_defects = 0
        
        for i, (x, y, w, h) in enumerate(bottle_boxes):
            print(f"\n处理瓶子 {i+1}...")
            
            # 展宽边距确保字符完整，注意防止越界
            pad_x, pad_y = 15, 15
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(test_img.shape[1], x + w + pad_x)
            y2 = min(test_img.shape[0], y + h + pad_y)
            
            # 提取ROI
            roi = test_img[y1:y2, x1:x2]
            
            # 检测单体缺陷
            defects, M = self.detect_defects_single(roi)
            
            if defects:
                color = (0, 0, 255)
                status = f"DEFECT ({len(defects)})"
                total_defects += len(defects)
            else:
                color = (0, 255, 0)
                status = "OK"
            
            # 在大图绘制当前瓶子的限界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(result_img, f"B{i+1}: {status}", (x1, max(20, y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 如果有缺陷且变换矩阵有效，绘制具体的断笔画位置
            if defects and M is not None:
                try:
                    for defect in defects:
                        dx, dy, dw, dh = defect['bbox']
                        
                        # 构造缺陷在模板中的四个角点
                        pts = np.float32([[dx, dy], [dx+dw, dy], 
                                         [dx+dw, dy+dh], [dx, dy+dh]]).reshape(-1, 1, 2)
                        
                        # 修改：因为 M 已经是 Template -> Test ROI 了，直接正向变换即可，不需要 np.linalg.inv(M)
                        transformed_pts = cv2.perspectiveTransform(pts, M)
                        
                        # 加上当前瓶子的截取偏移，转换到原大图坐标系
                        mapped_pts = np.int32(transformed_pts.reshape(-1, 2) + [x1, y1])
                        
                        # 绘制缺陷多边形轮廓
                        cv2.polylines(result_img, [mapped_pts], True, (0, 0, 255), 2)
                except Exception as e:
                    print(f"瓶子 {i+1} 的坐标变换失败: {e}")
        
        # 显示结果
        cv2.namedWindow("Multi-Bottle Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Bottle Detection", 1200, 800)
        cv2.imshow("Multi-Bottle Detection", result_img)
        
        # 保存结果
        cv2.imwrite("detection_result.jpg", result_img)
        
        # 打印统计信息
        self._print_statistics(len(bottle_boxes), total_defects)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return bottle_boxes
    
    def _print_statistics(self, bottle_count, defect_count):
        """
        打印检测统计信息
        """
        print("\n" + "="*60)
        print("针剂瓶字符缺陷检测报告")
        print("="*60)
        print(f"检测瓶子总数: {bottle_count}")
        print(f"累计检出断裂/缺失缺陷数量: {defect_count}")
        print("="*60)

# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = DefectDetector("C:\\Users\\ASUS\\Desktop\\test\\3\\template.jpg")
    
    # 检测多瓶子图像
    detector.detect_multiple_bottles("C:\\Users\\ASUS\\Desktop\\test\\3\\test.jpg")