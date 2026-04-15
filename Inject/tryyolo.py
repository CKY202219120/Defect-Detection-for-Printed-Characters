import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# =========================================================================
# 常规图像支持与预处理辅助函数
# =========================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_bgr(gray):
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def preprocess_image(img):
    """增强对比度、去噪、锐化，使特征更稳定"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def build_char_mask(template_gray):
    """构建字符区域掩膜（白色区域为字符）"""
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


# =========================================================================
# 模块一：YOLOv8 文本检测与区域合并
# =========================================================================

def detect_text_regions_by_yolo(model, img_path, conf_thr=0.15, merge_margin=60):
    """使用 YOLO 检测单个字符，并通过合并相近框形成完整的瓶子文字区域 ROI"""
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    # YOLO 预测所有类别寻找文字分布
    results = model.predict(source=img_path, conf=conf_thr, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        
    if not boxes:
        return []

    # 转为 [x1, y1, x2, y2] 方便拓展
    rects = []
    for (x, y, w, h) in boxes:
        rects.append([
            x - merge_margin//2, 
            y - merge_margin//2, 
            x + w + merge_margin//2, 
            y + h + merge_margin//2
        ])
    
    def merge_rects(r1, r2):
        if not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3]):
            return [min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3])]
        return None

    merged = True
    while merged:
        merged = False
        new_rects = []
        while rects:
            r = rects.pop(0)
            merged_with_any = False
            for i, other in enumerate(new_rects):
                res = merge_rects(r, other)
                if res is not None:
                    new_rects[i] = res
                    merged = True
                    merged_with_any = True
                    break
            if not merged_with_any:
                new_rects.append(r)
        rects = new_rects

    final_regions = []
    img_h, img_w = img.shape[:2]
    for r in rects:
        rx1 = max(0, r[0] + merge_margin//2)
        ry1 = max(0, r[1] + merge_margin//2)
        rx2 = min(img_w, r[2] - merge_margin//2)
        ry2 = min(img_h, r[3] - merge_margin//2)
        final_regions.append([rx1, ry1, rx2 - rx1, ry2 - ry1])

    return sorted(final_regions, key=lambda b: b[0])


# =========================================================================
# 模块二：SIFT 特征提取与骨架差分 (完美剥离自你的特征匹配代码)
# =========================================================================

def check_homography(H, template_shape, roi_shape):
    h_t, w_t = template_shape
    h_r, w_r = roi_shape
    det = np.linalg.det(H[:2, :2])
    if det < 0.25 or det > 4.0:
        return False
    if abs(H[2, 0]) > 0.01 or abs(H[2, 1]) > 0.01:
        return False
    pts_t = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    pts_r = cv2.perspectiveTransform(pts_t, H)
    def polygon_area(pts):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_ratio = polygon_area(pts_r) / (w_t * h_t)
    if area_ratio < 0.5 or area_ratio > 2.0:
        return False
    return True

def sift_register(template_gray, roi_gray, min_matches=8):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.04, edgeThreshold=15)
    kp_t, des_t = sift.detectAndCompute(template_gray, None)
    kp_r, des_r = sift.detectAndCompute(roi_gray, None)
    if des_t is None or des_r is None or len(kp_r) < min_matches:
        return None, None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    knn = flann.knnMatch(des_r, des_t, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]

    if len(good) < min_matches:
        return None, None

    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    
    if H is None or not check_homography(H, template_gray.shape, roi_gray.shape):
        return None, None

    h, w = template_gray.shape[:2]
    aligned = cv2.warpPerspective(roi_gray, H, (w, h))
    return H, aligned

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

def detect_defects_optimized(template_gray, aligned, char_mask, tol_pixels=2, min_area=5):
    blended_aligned = cv2.addWeighted(template_gray, 0.5, aligned, 0.5, 0)
    th_t = cv2.adaptiveThreshold(blended_aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    th_a = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    
    th_t = cv2.bitwise_and(th_t, th_t, mask=char_mask)
    th_a = cv2.bitwise_and(th_a, th_a, mask=char_mask)

    def clean_binary(binary_image):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
        mask = np.zeros_like(opened)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 20 < cv2.contourArea(cnt) < 5000:
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
# 模块三：混合调度主函数
# =========================================================================

def hybrid_run(yolo_model_path, template_path, test_path, out_dir):
    ensure_dir(out_dir)

    print("-> 正在加载 YOLO 模型...")
    model = YOLO(yolo_model_path)
    
    template_bgr = cv2.imread(template_path)
    test_bgr = cv2.imread(test_path)
    if template_bgr is None or test_bgr is None:
        print("错误: 无法读取模板或测试图像")
        return

    print("步骤1: [YOLOv8 定位] 检测并合并寻找完整字符区...")
    boxes = detect_text_regions_by_yolo(model, test_path, conf_thr=0.15, merge_margin=60)
    
    if not boxes:
        print("YOLO 未检测到任何字符区，测试退出。")
        return
    print(f"-> 成功定位到 {len(boxes)} 个连续文本块。")

    template_gray_raw = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    test_gray_raw = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    template_gray = preprocess_image(template_gray_raw)
    test_gray = preprocess_image(test_gray_raw)
    char_mask = build_char_mask(template_gray)

    overview = test_bgr.copy()
    defect_cnt, pass_cnt, uncertain_cnt = 0, 0, 0

    print("步骤2: [SIFT+形态学] 进行瓶子细节高精度比对...")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        pad_x, pad_y = 15, 15
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(test_gray.shape[1], x + w + pad_x), min(test_gray.shape[0], y + h + pad_y)
        roi_gray = test_gray[y1:y2, x1:x2]

        H, aligned = sift_register(template_gray, roi_gray, min_matches=8)

        if H is None or aligned is None:
            uncertain_cnt += 1
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.putText(overview, f"#{i} UNCERTAIN", (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            continue

        defects, debug_imgs = detect_defects_optimized(template_gray, aligned, char_mask, tol_pixels=2)

        if defects:
            defect_cnt += 1
            color, label = (0,0,255), f"#{i} DEFECT ({len(defects)})"
            H_inv = np.linalg.inv(H)
            for poly in map_contours_to_original(defects, H_inv, x1, y1):
                cv2.polylines(overview, [poly], True, (0,0,255), 2)
        else:
            pass_cnt += 1
            color, label = (0,200,0), f"#{i} PASS"

        cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overview, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 导出 Debug 图
        tpl_show = put_title(to_bgr(template_gray), "Template")
        ali_show = put_title(to_bgr(aligned), "Aligned")
        blend_show = put_title(to_bgr(cv2.addWeighted(template_gray, 0.4, aligned, 0.6, 0)), "50:50 Blend")
        diff_show = put_title(to_bgr(cv2.absdiff(template_gray, cv2.addWeighted(template_gray, 0.4, aligned, 0.6, 0))), "Diff")
        cv2.imwrite(os.path.join(out_dir, f"bottle_{i:02d}_debug01_base.jpg"), hstack_resize([tpl_show, ali_show, blend_show, diff_show], h=200))
        
        cv2.imwrite(os.path.join(out_dir, f"bottle_{i:02d}_debug02_skeleton.jpg"), hstack_resize([
            put_title(to_bgr(debug_imgs['bin_template']), "Bin(Temp)"),
            put_title(to_bgr(debug_imgs['skeleton']), "Skel(Temp)"),
            put_title(to_bgr(debug_imgs['bin_aligned']), "Bin(Align)"),
            put_title(to_bgr(debug_imgs['dilated_aligned']), "Dilate(Align)"),
            put_title(to_bgr(debug_imgs['diff']), "Defect Diff")
        ], h=200))

    summary = f"Total:{len(boxes)}  PASS:{pass_cnt}  DEFECT:{defect_cnt}  UNCERTAIN:{uncertain_cnt}"
    cv2.rectangle(overview, (0,0), (overview.shape[1], 32), (245,245,245), -1)
    cv2.putText(overview, summary, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2, cv2.LINE_AA)
    
    out_path = os.path.join(out_dir, "result_overview.jpg")
    cv2.imwrite(out_path, overview)
    print(f"所有分析完成！结果已保存至: {out_path}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + SIFT形态学缺陷混合检测")
    parser.add_argument("--yolo", default=r"finetune\yolov8m_final.pt", help="YOLO模型路径")
    parser.add_argument("--template", required=True, help="完美无缺陷的基准模板图片路径")
    parser.add_argument("--test", required=True, help="需要检测的带针剂瓶子图像路径")
    parser.add_argument("--out", default="hybrid_output", help="分析结果输出目录")
    
    args = parser.parse_args()
    hybrid_run(args.yolo, args.template, args.test, args.out)