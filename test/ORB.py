import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_and_match_orb(template_path, test_path, nfeatures=5000):
    """
    基于ORB算子的特征点提取与匹配
    :param template_path: 模板图像路径（单个清晰的针剂瓶打印字符）
    :param test_path: 测试图像路径（多个并列排放的待检测针剂瓶）
    :param nfeatures: ORB提取的最大特征点数量，针对多个目标可适当调大
    """
    # 1. 读取图像 (灰度模式)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template_img is None or test_img is None:
        print("错误: 无法读取图像，请检查文件路径是否正确。")
        return

    # 2. 初始化ORB检测器
    # 参数调整建议：可调节 nfeatures 获取更多特征点; 调整 scaleFactor 和 nlevels 适应不同大小的字符
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15)

    # 3. 寻找关键点(Keypoints)和计算描述子(Descriptors)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(test_img, None)

    if des1 is None or des2 is None:
        print("未能提取到足够的特征点。")
        return

    print(f"模板图提取特征点数: {len(kp1)}")
    print(f"测试图提取特征点数: {len(kp2)}")

    # 4. 特征匹配 - 初始化蛮力匹配器 (Brute-Force Matcher) 和汉明距离 (Hamming distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # 5. 使用K近邻(KNN)算法匹配，k=2表示寻找最近的两个匹配点
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # 6. 使用 Lowe's ratio test 过滤误匹配点
    good_matches = []
    ratio_threshold = 0.75  # 阈值越小，匹配要求越严格 (通常在0.7~0.8之间)
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    print(f"经过比率测试后的有效匹配点对数: {len(good_matches)}")

    # 7. 绘制匹配结果
    # flags=2 表示只绘制成功匹配对关联的关键点，不绘制孤立点和未匹配的点
    matched_img = cv2.drawMatches(
        template_img, kp1, 
        test_img, kp2, 
        good_matches, None, 
        matchColor=(0, 255, 0),       # 成功匹配的连线颜色为绿色
        singlePointColor=(255, 0, 0), # 关键点颜色为蓝色
        flags=2
    )

    # 8. 可视化显示
    plt.figure(figsize=(15, 8))
    plt.title(f"ORB Feature Matching: {len(good_matches)} Good Matches")
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def detect_defects_orb(template_path, test_path):
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. 提取特征并匹配 (与您现有代码相同)
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(test_img, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 2. 从良好匹配中提取对应的点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 3. 寻找单应性矩阵 (如果图里只有一个目标，这步很有效；如果有多个，需要循环或聚类)
    # MIN_MATCH_COUNT 至少需要 4 个点
    if len(good_matches) > 10: 
        # RANSAC可以剔除异常点，找到主要的那个匹配对象的变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # 4. 将模板图对齐到测试图视角
            h, w = template_img.shape
            aligned_template = cv2.warpPerspective(template_img, M, (test_img.shape[1], test_img.shape[0]))
            
            # 5. 计算差异 (在此之前可做一下模糊降噪)
            # 仅在有对齐图案的区域内做对比 (可以使用掩码)
            diff = cv2.absdiff(test_img, aligned_template)
            
            # 6. 阈值化找出缺陷
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            
            # 形态学操作去除小噪点
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 标注缺陷 (画红框等)
            # ...
    else:
        print("Not enough matches are found")

if __name__ == "__main__":
    # TODO: 请在此处替换为您本地照片的绝对路径或相对路径
    TEMPLATE_IMAGE = r"C:\Users\ASUS\Desktop\test\3\template.jpg"   # 模板图：单个清楚打印的样本
    TEST_IMAGE = r"C:\Users\ASUS\Desktop\test\3\template2.jpg"           # 测试图：多个并列排放的待检测针剂瓶
    
    extract_and_match_orb(TEMPLATE_IMAGE, TEST_IMAGE)