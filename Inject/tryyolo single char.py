import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# =========================================================================
# 辅助函数：图像处理与骨架提取
# =========================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_bgr(gray):
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def preprocess_image(img):
    """增强对比度、去噪、锐化"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def build_char_mask(template_gray):
    """构建单字符区域的可用掩膜"""
    _, inv = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def put_title(img, text, color=(20,20,220)):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w,28), (245,245,245), -1)
    cv2.putText(out, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out

def hstack_resize(images, h=260):
    resized = []
    for im in images:
        scale = h / im.shape[0]
        w = max(1, int(im.shape[1] * scale))
        resized.append(cv2.resize(im, (w, h)))
    return cv2.hconcat(resized)

def extract_skeleton(binary_image):
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(binary_image)
    skeleton = np.zeros_like(binary_image)
    img = binary_image.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, temp))
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skeleton


# =========================================================================
# 模块一：模板库加载与最匹配搜索
# =========================================================================

def load_template_library(template_dir):
    """加载模板文件夹中的所有基准字符图片"""
    library = {}
    if not os.path.exists(template_dir):
         print(f"模板库目录不存在: {template_dir}")
         return library
         
    for fname in os.listdir(template_dir):
        if fname.lower().endswith(('.jpg', '.png', '.bmp')):
            path = os.path.join(template_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 以文件名（不含后缀）为键名，存入预处理后的灰度图
                name = os.path.splitext(fname)[0]
                library[name] = preprocess_image(img)
    print(f"-> 成功加载了 {len(library)} 个单字符模板模板。")
    return library

def find_best_template(roi_gray, template_library):
    """
    通过将 ROI Resize 后与库中所有模板进行结构相似度或跨相关比较，
    识别当前 YOLO 框出的字符到底对应哪个模板。
    """
    best_name = None
    best_score = -1.0
    best_template = None
    
    # 因为此时图像已经只有这一个字符了，用 Resize 后的直接相关性判断最准
    for name, tpl in template_library.items():
        th, tw = tpl.shape
        # 将待测 ROI resize 到与该测试模板同样大
        resized_roi = cv2.resize(roi_gray, (tw, th))
        # 全局相关性匹配法
        res = cv2.matchTemplate(resized_roi, tpl, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        
        if score > best_score:
            best_score = score
            best_name = name
            best_template = tpl
            
    return best_name, best_score, best_template


# =========================================================================
# 模块二：SIFT 特征提取与骨架差分 (完全适配单字符级别)
# =========================================================================
# (注意：因为已经是单字符级别对比，单应性检查要求可适当放宽甚至去除)

def align_single_character(template_gray, roi_gray, min_matches=6):
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)
    kp_t, des_t = sift.detectAndCompute(template_gray, None)
    kp_r, des_r = sift.detectAndCompute(roi_gray, None)
    if des_t is None or des_r is None or len(kp_r) < min_matches:
        return None, None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des_r, des_t, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < min_matches:
        return None, None

    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # 由于单字符目标很小，仿射变换(Affine)往往比单应性(Homography)更坚固稳定
    # 但我们优先尝试 FindHomography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=500, confidence=0.95)
    
    if H is None:
        return None, None

    # 直接使用带有严格限制的刚体/相似仿射变换（平移+旋转+等比缩放），剥夺透视倾斜和非等宽拉伸的自由度
    A, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if A is None:
        return None, None

    # 为了兼容你后续代码里用的 H_inv = np.linalg.inv(H) 和 perspectiveTransform，
    # 我们把 2x3 的仿射矩阵 A 转为标准的 3x3 单应性矩阵 H
    H = np.eye(3, dtype=np.float32)
    H[0:2, :] = A
    
    # 增加额外的缩放合理性检查（可选）：如果缩放比例过于离谱(比如放大 3倍 或者缩小到 0.3倍)，则拒绝对齐
    scale = np.sqrt(A[0, 0]**2 + A[0, 1]**2)
    if scale > 2.0 or scale < 0.5:
        return None, None

    h, w = template_gray.shape[:2]
    aligned = cv2.warpPerspective(roi_gray, H, (w, h))
    return H, aligned

def detect_defects_optimized(template_gray, aligned, char_mask, tol_pixels=2, min_area=3):
    blended_aligned = cv2.addWeighted(template_gray, 0.5, aligned, 0.5, 0)
    th_t = cv2.adaptiveThreshold(blended_aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    th_a = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    th_t = cv2.bitwise_and(th_t, th_t, mask=char_mask)
    th_a = cv2.bitwise_and(th_a, th_a, mask=char_mask)

    def clean_binary(binary_image):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
        mask = np.zeros_like(opened)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 5 < cv2.contourArea(cnt): # 放宽上限，因为现在是单字符
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return cv2.bitwise_and(binary_image, mask)

    th_t = clean_binary(th_t)
    th_a = clean_binary(th_a)
    skeleton_t = extract_skeleton(th_t)

    k_size = tol_pixels * 2 + 1
    dilated_a = cv2.dilate(th_a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
    diff = cv2.bitwise_and(skeleton_t, cv2.bitwise_not(dilated_a))
    diff_cleaned = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    
    contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
            
    debug_imgs = {'bin_template': th_t, 'bin_aligned': th_a, 'skeleton': skeleton_t, 'dilated_aligned': dilated_a, 'diff': diff_cleaned}
    return defects, debug_imgs

def map_contours_to_original(contours, H_inv, roi_x, roi_y):
    mapped = []
    for cnt in contours:
        cnt_roi = cv2.perspectiveTransform(cnt.astype(np.float32).reshape(-1,1,2), H_inv)
        mapped.append((cnt_roi + np.array([[[roi_x, roi_y]]], dtype=np.float32)).astype(np.int32))
    return mapped


# =========================================================================
# 模块三：主检测流
# =========================================================================

def single_char_hybrid_run(yolo_model_path, templates_dir, test_path, out_dir):
    ensure_dir(out_dir)

    # 1. 挂载 YOLO 及模板库
    print("-> 正在加载 YOLO 模型...")
    model = YOLO(yolo_model_path)
    
    print("-> 正在读取单字符基准模板库...")
    template_library = load_template_library(templates_dir)
    if not template_library:
        print("错误: 模板库为空或未找到目录。请将截好的单字符标准图放在:", templates_dir)
        return

    test_bgr = cv2.imread(test_path)
    if test_bgr is None:
        print(f"错误: 无法读取测试图像 {test_path}")
        return
        
    test_gray_raw = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = preprocess_image(test_gray_raw)

    # 2. YOLO 获取单字符级的 ROI
    print("步骤1: [YOLOv8 定位] 搜索全图中每一个单个字符...")
    results = model.predict(source=test_path, conf=0.15, verbose=False)
    
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append([x1, y1, x2-x1, y2-y1])
        
    if not boxes:
        print("YOLO 未检测到任何字符。")
        return
        
    # 按从左到右，从上到下午排序（方便查看输出）
    boxes = sorted(boxes, key=lambda b: (b[1]//20, b[0]))
    print(f"-> 成功提取到 {len(boxes)} 个单字符包围盒。")

    overview = test_bgr.copy()
    defect_cnt, pass_cnt, uncertain_cnt = 0, 0, 0

    print("步骤2: [匹配与差分] 智能指派模板并做像素级异常诊断...")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        # 对 YOLO 框做极微小的 expand，确保字符不要卡边
        pad = 4
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(test_gray.shape[1], x + w + pad), min(test_gray.shape[0], y + h + pad)
        
        roi_gray = test_gray[y1:y2, x1:x2]

        # 如果这个字太小（比如噪点被误检）
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        # ---------------- A. 到模板库里找最像的基准模板 ----------------
        best_name, confidence, best_template_gray = find_best_template(roi_gray, template_library)
        
        # 信心度太低说明它什么都不是（或许是杂质被误检）
        if confidence < 0.3:
            color = (150, 150, 150)
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 1)
            cv2.putText(overview, "noise", (x1, max(10, y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            continue

        # ---------------- B. 生成该特定字符模板的 Mask ----------------
        char_mask = build_char_mask(best_template_gray)

        # ---------------- C. SIFT 精确对齐 ----------------
        H, aligned = align_single_character(best_template_gray, roi_gray, min_matches=5)

        if H is None or aligned is None:
            uncertain_cnt += 1
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0,255,255), 1)
            # 标注它被猜测是什么字符，但是对齐失败了
            cv2.putText(overview, f"{best_name}?", (x1, max(10, y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            continue

        # ---------------- D. 骨架过滤差异比对检测 ----------------
        defects, debug_imgs = detect_defects_optimized(best_template_gray, aligned, char_mask, tol_pixels=2)

        if defects:
            defect_cnt += 1
            color, label = (0,0,255), f"{best_name}(err)"
            # 把模板上的断笔区域直接用多边形画出来
            H_inv = np.linalg.inv(H)
            for poly in map_contours_to_original(defects, H_inv, x1, y1):
                cv2.polylines(overview, [poly], True, (0,0,255), 1)
        else:
            pass_cnt += 1
            color, label = (0,200,0), f"{best_name}"

        # 画本单字的概览框和字号判定
        cv2.rectangle(overview, (x1, y1), (x2, y2), color, 1)
        cv2.putText(overview, label, (x1, max(12, y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 如果发现缺陷，导出它的专属差分对比小图
        if defects:
            debug_panel = hstack_resize([
                put_title(to_bgr(best_template_gray), f"Tpl({best_name})"),
                put_title(to_bgr(aligned), "Aligned ROI"),
                put_title(to_bgr(debug_imgs['skeleton']), "Base Skel"),
                put_title(to_bgr(debug_imgs['dilated_aligned']), "Dilated ROI"),
                put_title(to_bgr(debug_imgs['diff']),"Diff/Defect")
            ], h=120)
            cv2.imwrite(os.path.join(out_dir, f"char_{i:03d}_err_debug.jpg"), debug_panel)

    # 导出总分析图
    summary = f"Total:{len(boxes)} | PASS:{pass_cnt} | DEFECT:{defect_cnt} | UNCERTAIN:{uncertain_cnt}"
    cv2.rectangle(overview, (0,0), (overview.shape[1], 25), (245,245,245), -1)
    cv2.putText(overview, summary, (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 2, cv2.LINE_AA)
    
    out_path = os.path.join(out_dir, "result_overview.jpg")
    cv2.imwrite(out_path, overview)
    print(f"所有分析完成！结果已保存至: {out_path}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单字符级别 - YOLO定位&模板最匹配检索&骨架缺陷检测")
    parser.add_argument("--yolo", default=r"finetune\yolov8m_final.pt", help="YOLO模型路径")
    parser.add_argument("--templates_dir", required=True, help="存放各个预切好标准字符图片的文件夹路径")
    parser.add_argument("--test", required=True, help="需要检测的被测图（原图）")
    parser.add_argument("--out", default="character_level_output", help="分析结果输出目录")
    
    args = parser.parse_args()
    single_char_hybrid_run(args.yolo, args.templates_dir, args.test, args.out)


# D:/minicoda/python.exe "tryyolo single char.py" --templates_dir "C:\Users\ASUS\Desktop\Inject\char_templates" --test "C:\Users\ASUS\Desktop\test\3\test2.jpg" --out "单字符_测试输出"
# D:/minicoda/python.exe "tryyolo single char.py" --templates_dir "C:\Users\ASUS\Desktop\Inject\char_templates2" --test "C:\Users\ASUS\Desktop\test\4\test.jpg" --out "单字符_测试输出"
# D:/minicoda/python.exe "tryyolo single char.py" --templates_dir "C:\Users\ASUS\Desktop\Inject\char_templates3" --test "C:\Users\ASUS\Desktop\test\2\test.jpg" --out "单字符_测试输出"