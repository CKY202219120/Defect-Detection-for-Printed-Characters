import os
import cv2
import argparse
import numpy as np
import time  # <--- 新增导入

cv2.ocl.setUseOpenCL(False)  # 强制关闭 OpenCV 的 OpenCL 加速

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_bgr(gray):
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def preprocess_image(img):
    """增强对比度、去噪、锐化，使特征更稳定"""
    # CLAHE增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # 双边滤波保留边缘去噪
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # 锐化
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def build_char_mask(template_gray):
    """构建字符区域掩膜（白色区域为字符）"""
    # Otsu二值化得到字符（黑色）背景（白色），取反使字符为白色
    _, inv = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 开运算去除小噪点，再膨胀覆盖字符区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_bottles_by_template(test_gray, template_gray, match_thr=0.45):
    """
    模板粗定位，返回每个瓶子的边界框 [x,y,w,h]
    """
    h, w = template_gray.shape
    # 使用原始模板（未预处理）进行匹配，避免预处理改变外观
    res = cv2.matchTemplate(test_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= match_thr)

    rectangles = []
    for pt in zip(*loc[::-1]):  # 交换x,y顺序
        rectangles.append([int(pt[0]), int(pt[1]), w, h])

    # 合并重叠框 (groupThreshold=1 表示至少1个框才保留，eps控制合并的IoU阈值)
    boxes, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)
    # 按x坐标排序（从左到右）
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes
def check_homography(H, template_shape, roi_shape):
    """
    验证单应矩阵是否合理
    H: 从ROI到模板的变换矩阵
    template_shape: (h, w)
    roi_shape: (h_roi, w_roi)
    返回 True 表示合理，False 表示异常
    """
    h_t, w_t = template_shape
    h_r, w_r = roi_shape

    # 1. 检查缩放比例（通过行列式）
    det = np.linalg.det(H[:2, :2])  # 只考虑线性部分
    if det < 0.25 or det > 4.0:
        return False

    # 2. 检查透视分量大小
    if abs(H[2, 0]) > 0.01 or abs(H[2, 1]) > 0.01:
        # 透视分量过大，可能导致拉伸
        return False

    # 3. 将模板的四个角点变换到ROI空间，检查变形程度
    pts_t = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    pts_r = cv2.perspectiveTransform(pts_t, H)  # 变换到ROI坐标系

    # 计算变换后四边形的面积（使用鞋带公式）
    def polygon_area(pts):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    area_ratio = polygon_area(pts_r) / (w_t * h_t)
    if area_ratio < 0.5 or area_ratio > 2.0:
        return False

    # 4. 检查变换后的点是否大部分在ROI内部（可选）
    # 如果所有点都严重偏离ROI，则不可靠
    pts_r = pts_r.reshape(-1, 2)
    if not (np.all(pts_r[:, 0] > -0.5 * w_r) and np.all(pts_r[:, 0] < 1.5 * w_r) and
            np.all(pts_r[:, 1] > -0.5 * h_r) and np.all(pts_r[:, 1] < 1.5 * h_r)):
        return False

    return True

def sift_register(template_gray, roi_gray, min_matches=8):
    """
    SIFT配准：计算从ROI到模板的单应矩阵H，并对齐ROI到模板尺寸
    返回: H, aligned 或 (None, None)
    """
    if not hasattr(cv2, 'SIFT_create'):
        raise RuntimeError("OpenCV 缺少SIFT模块，请安装 opencv-contrib-python")

    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.04, edgeThreshold=15)
    kp_t, des_t = sift.detectAndCompute(template_gray, None)
    kp_r, des_r = sift.detectAndCompute(roi_gray, None)

    if des_t is None or des_r is None or len(kp_r) < min_matches:
        return None, None

    # FLANN匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn = flann.knnMatch(des_r, des_t, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            # 更严格的比率测试
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < min_matches:
        return None, None

    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1,1,2)   # ROI点
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1,1,2)   # 模板点

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    if H is None:
        return None, None

    # 验证H的合理性
    if not check_homography(H, template_gray.shape, roi_gray.shape):
        return None, None

    h, w = template_gray.shape[:2]
    aligned = cv2.warpPerspective(roi_gray, H, (w, h))
    return H, aligned

def orb_register(template_gray, roi_gray, min_matches=8):
    """
    ORB配准：计算从ROI到模板的单应矩阵H，并对齐ROI到模板尺寸
    返回: H, aligned 或 (None, None)
    """
    # 1. 创建 ORB 提取器，最大特征点数设为3000保证有足够的点匹配
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    
    kp_t, des_t = orb.detectAndCompute(template_gray, None)
    kp_r, des_r = orb.detectAndCompute(roi_gray, None)

    # 防止未提取到足够的特征点
    if des_t is None or des_r is None or len(kp_r) < min_matches or len(kp_t) < min_matches:
        return None, None

    # 2. 暴力匹配器 (Brute-Force) + 汉明距离 (NORM_HAMMING适用于二进制描述子)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_r, des_t, k=2)

    # 3. 比率测试 (Lowe's Ratio Test)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            # ORB 的二进制特征距离容差通常设在 0.75 到 0.8 之间
            if m.distance < 0.85 * n.distance:
                good.append(m)

    if len(good) < min_matches:
        return None, None

    # 4. 获取匹配点的主坐标
    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1,1,2)   # ROI点
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1,1,2)   # 模板点

    # 5. RANSAC 求解单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    if H is None:
        return None, None

    # 验证 H 的合理性 (如果发生严重畸变则丢弃)
    if not check_homography(H, template_gray.shape, roi_gray.shape):
        return None, None

    # 6. 利用 H 矩阵将 ROI 进行仿射变换对齐到模板尺寸
    h, w = template_gray.shape[:2]
    aligned = cv2.warpPerspective(roi_gray, H, (w, h))
    return H, aligned

def detect_defects(template_gray, aligned_gray, char_mask, min_area=15, max_area=2500):
    """
    检测对齐图像相对于模板的缺陷（断裂、缺失）
    返回：缺陷轮廓列表（在模板坐标系下）
    """
    # 轻微高斯模糊吸收对齐误差
    t = cv2.GaussianBlur(template_gray, (3,3), 0)
    a = cv2.GaussianBlur(aligned_gray, (3,3), 0)

    diff = cv2.absdiff(t, a)
    # 仅关注字符区域
    diff = cv2.bitwise_and(diff, diff, mask=char_mask)

    # Otsu自动阈值
    _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学去噪、连接断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            defects.append(cnt)
    return defects

def map_contours_to_original(contours, H_inv, roi_x, roi_y):
    """
    将模板坐标系下的轮廓映射回原图
    contours: 列表，每个元素为(N,1,2)的轮廓点
    H_inv: 模板->ROI的单应矩阵 (即H的逆)
    roi_x, roi_y: ROI在原图中的左上角坐标
    返回：映射后的轮廓列表（整数坐标）
    """
    mapped = []
    for cnt in contours:
        # 轮廓点变形
        cnt_float = cnt.astype(np.float32).reshape(-1,1,2)
        cnt_roi = cv2.perspectiveTransform(cnt_float, H_inv)   # 转到ROI坐标系
        cnt_full = cnt_roi + np.array([[[roi_x, roi_y]]], dtype=np.float32)  # 转到全图
        mapped.append(cnt_full.astype(np.int32))
    return mapped

def put_title(img, text, color=(20,20,220)):
    """在图像顶部添加标题栏"""
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w,28), (245,245,245), -1)
    cv2.putText(out, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out

def hstack_resize(images, h=260):
    """将多个图像水平拼接并统一高度"""
    resized = []
    for im in images:
        scale = h / im.shape[0]
        w = max(1, int(im.shape[1] * scale))
        resized.append(cv2.resize(im, (w, h)))
    return cv2.hconcat(resized)

def extract_skeleton(binary_image):
    """
    提取二值图像的骨架
    如果安装了 opencv-contrib-python，使用 ximgproc 模块效果更好
    否则采用形态学方法进行简化提取
    """
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(binary_image)
    
    # 降级方案：经典形态学骨架提取
    skeleton = np.zeros_like(binary_image)
    img = binary_image.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skeleton

def detect_defects_optimized(template_gray, aligned, char_mask, tol_pixels=2, min_area=5):
    """
    基于骨架提取与邻域容差的字符缺陷检测，并包含反光噪点过滤机制
    """

    # ====== 核心修改：50:50 图像融合作为新模板 ======
    # 把原模板和当前侧视图(aligned)按照 0.5和0.5 的权重融合成一张新图
    blended_aligned = cv2.addWeighted(template_gray, 0.35, aligned, 0.75, 0)
    
    # 【按照你的需求】：将融合图(blended_aligned)作为新模板，将原侧视图(aligned)作为对比图
    th_t = cv2.adaptiveThreshold(blended_aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 12)
    th_a = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 12)
    # ================================================

    # 仅保留字符区域掩膜内的内容
    th_t = cv2.bitwise_and(th_t, th_t, mask=char_mask)
    th_a = cv2.bitwise_and(th_a, th_a, mask=char_mask)

    # ================= 新增：二值图像杂质清洗 =================
    def clean_binary(binary_image):
        # (1) 形态学开运算：先腐蚀后膨胀。
        # 能直接抹除游离的极细小散落像素带，打断相连的细丝反光。
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
        
        # (2) 连通域过滤：剔除不是文字块的形状
        cleaned = np.zeros_like(opened)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 面积过滤：太小的是灰尘噪点（<3），太大的是成片高光反光块（>600）
            # 注意：如果你的字符连在一起非常大，可以把上限 600 调大一点
            if 20 < area < 3000:
                # 形状比例过滤 (可选)：你可以通过外接矩形宽高比，过滤极端的细长反光条
                # x, y, w, h = cv2.boundingRect(cnt)
                # if w/h > 8 or h/w > 8: continue 
                
                cv2.drawContours(cleaned, [cnt], -1, 255, -1)
        return cleaned

    # 对模板和待测图都做一次清洗
    th_t = clean_binary(th_t)
    th_a = clean_binary(th_a)
    # ========================================================

    # 2. 提取纯净模板文字的骨架
    skeleton_t = extract_skeleton(th_t)

    # 3. 对目标图像进行膨胀（容差匹配）
    kernel_size = tol_pixels * 2 + 1
    tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_a = cv2.dilate(th_a, tol_kernel)

    # 4. 寻找缺失缺陷
    diff = cv2.bitwise_and(skeleton_t, cv2.bitwise_not(dilated_a))

    # 5. 形态学清理细微噪点
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    diff_cleaned = cv2.dilate(diff, clean_kernel, iterations=1)
    
    # 6. 查找缺陷轮廓
    contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defects = []
    for cnt in contours:
        # 这个 min_area 决定了多大的断墨才算真正的缺陷（目前默认5像素）
        area = cv2.contourArea(cnt)
        if area >= min_area:
            defects.append(cnt)
            
    # 打包中间过程返回
    debug_imgs = {
        'bin_template': th_t,
        'bin_aligned': th_a,
        'skeleton': skeleton_t,
        'dilated_aligned': dilated_a,
        'diff': diff_cleaned
    }
            
    return defects, debug_imgs

def run(template_path, test_path, out_dir, match_thr=0.45):
    ensure_dir(out_dir)

    # 读取图像
    template_bgr = cv2.imread(template_path)
    test_bgr = cv2.imread(test_path)
    if template_bgr is None or test_bgr is None:
        raise FileNotFoundError("无法读取图像，请检查路径")

    # 转为灰度
    template_gray_raw = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    test_gray_raw = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    # 预处理（用于SIFT和缺陷检测）
    template_gray = preprocess_image(template_gray_raw)
    test_gray = preprocess_image(test_gray_raw)

    # 构建字符掩膜
    char_mask = build_char_mask(template_gray)

    # 1. 多瓶定位（使用原始灰度图，避免预处理影响匹配）
    boxes = detect_bottles_by_template(test_gray_raw, template_gray_raw, match_thr)
    if len(boxes) == 0:
        print("未检测到任何瓶子，请降低 match_thr 或检查图像")
        return

    print(f"检测到 {len(boxes)} 个候选瓶体")

    # 准备总览图
    overview = test_bgr.copy()
    defect_cnt = 0
    pass_cnt = 0
    uncertain_cnt = 0

    total_register_time = 0.0  # <--- 新增：记录总配准时间

    for i, (x, y, w, h) in enumerate(boxes, 1):
        # 提取ROI（可适当扩大边界以包含完整字符）
        pad_x, pad_y = 15, 15
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(test_gray.shape[1], x + w + pad_x)
        y2 = min(test_gray.shape[0], y + h + pad_y)
        roi_gray = test_gray[y1:y2, x1:x2]

        # --- 新增计时开始 ---
        start_time = time.time()
        
        # ORB配准
        H, aligned = orb_register(template_gray, roi_gray, min_matches=8)

        # --- 新增计时结束 ---
        end_time = time.time()
        register_time = end_time - start_time
        total_register_time += register_time
        print(f"瓶子 #{i} ORB 配准耗时: {register_time * 1000:.2f} ms")

        if H is None or aligned is None:
            uncertain_cnt += 1
            color = (0,255,255)
            label = f"#{i} UNCERTAIN"
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overview, label, (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            continue

        # 缺陷检测，并获取所有的中间调试图
        defects, debug_imgs = detect_defects_optimized(template_gray, aligned, char_mask, tol_pixels=2)

        if defects:
            defect_cnt += 1
            color = (0,0,255)
            label = f"#{i} DEFECT ({len(defects)})"
        else:
            pass_cnt += 1
            color = (0,200,0)
            label = f"#{i} PASS"

        # 在总览图上绘制矩形和标签
        cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overview, label, (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if defects:
            H_inv = np.linalg.inv(H)
            mapped = map_contours_to_original(defects, H_inv, x1, y1)
            for poly in mapped:
                cv2.polylines(overview, [poly], True, (0,0,255), 2)

        # ---------------- 调试图像保存部分 ----------------
        # 1. 常规原图对比区
        tpl_show = put_title(to_bgr(template_gray), "Template")
        ali_show = put_title(to_bgr(aligned), "Aligned")
        
        # 增加展示 50:50 融合效果
        blended_show = cv2.addWeighted(template_gray, 0.5, aligned, 0.5, 0)
        diff_show = put_title(to_bgr(cv2.absdiff(template_gray, blended_show)), "Blend Diff")
        
        panel1 = hstack_resize([tpl_show, ali_show, put_title(to_bgr(blended_show), "50:50 Blend"), diff_show], h=200)
        cv2.imwrite(os.path.join(out_dir, f"bottle_{i:02d}_debug01_base.jpg"), panel1)
        
        # 2. 骨架特征提取过程区
        bin_t_show = put_title(to_bgr(debug_imgs['bin_template']), "Binarized(Temp)")
        skel_show = put_title(to_bgr(debug_imgs['skeleton']), "Skeleton(Temp)")
        bin_a_show = put_title(to_bgr(debug_imgs['bin_aligned']), "Binarized(Align)")
        dilate_a_show = put_title(to_bgr(debug_imgs['dilated_aligned']), "Dilated(Align)")
        def_diff_show = put_title(to_bgr(debug_imgs['diff']), "Defect Diff")
        
        panel2 = hstack_resize([bin_t_show, skel_show, bin_a_show, dilate_a_show, def_diff_show], h=200)
        cv2.imwrite(os.path.join(out_dir, f"bottle_{i:02d}_debug02_skeleton.jpg"), panel2)
        
        # ===== 后面部分保持不变 =====

    # 总览图添加统计信息
    avg_time = (total_register_time / len(boxes)) * 1000 if len(boxes) > 0 else 0
    summary = f"Total:{len(boxes)} PASS:{pass_cnt} DEF:{defect_cnt} UNC:{uncertain_cnt}"
    time_summary = f"ORB Avg Time: {avg_time:.2f} ms/bottle" # <--- 新增
    
    cv2.rectangle(overview, (0,0), (overview.shape[1], 55), (245,245,245), -1)
    cv2.putText(overview, summary, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2, cv2.LINE_AA)
    cv2.putText(overview, time_summary, (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150,30,30), 2, cv2.LINE_AA) # <--- 新增

    out_path = os.path.join(out_dir, "result_overview.jpg")
    cv2.imwrite(out_path, overview)
    print(f"结果已保存至: {out_path}")
    print(summary)
    print(time_summary) # <--- 打印平均耗时

def main():
    parser = argparse.ArgumentParser(description="针剂瓶打印字符缺陷检测")
    parser.add_argument("--template", required=True, help="模板图像路径（单个清晰瓶子）")
    parser.add_argument("--test", required=True, help="测试图像路径（多个并排瓶子）")
    parser.add_argument("--out", default="output", help="输出目录")
    parser.add_argument("--match_thr", type=float, default=0.45,
                        help="模板匹配阈值 (0~1)，越低越容易检测到瓶子，但可能引入误检")
    args = parser.parse_args()

    run(args.template, args.test, args.out, args.match_thr)

if __name__ == "__main__":
    main()

# D:/minicoda/python.exe "c:/Users/ASUS/Desktop/test/ORB skeleton filter addweight.py" --template "C:/Users/ASUS/Desktop/test/3/template.jpg" --test "C:/Users/ASUS/Desktop/test/3/test.jpg" --out "C:/Users/ASUS/Desktop/test/3/output" --match_thr 0.36   
# D:/minicoda/python.exe "c:/Users/ASUS/Desktop/test/ORB skeleton filter addweight.py" --template "C:/Users/ASUS/Desktop/test/4/template.jpg" --test "C:/Users/ASUS/Desktop/test/4/test.jpg" --out "C:/Users/ASUS/Desktop/test/4/output" --match_thr 0.2   
# D:/minicoda/python.exe "c:/Users/ASUS/Desktop/test/ORB skeleton filter addweight.py" --template "C:/Users/ASUS/Desktop/test/2/template.jpg" --test "C:/Users/ASUS/Desktop/test/2/test.jpg" --out "C:/Users/ASUS/Desktop/test/2/output" --match_thr 0.45   
# D:/minicoda/python.exe "c:/Users/ASUS/Desktop/test/ORB skeleton filter addweight.py" --template "C:/Users/ASUS/Desktop/test/1/template.jpg" --test "C:/Users/ASUS/Desktop/test/1/test.jpg" --out "C:/Users/ASUS/Desktop/test/1/output" --match_thr 0.3

