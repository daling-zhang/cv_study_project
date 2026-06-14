import cv2
import numpy as np

# ---------------------- 1. 读取原图 ----------------------
img_path = "images/doc.jpg"
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败，请检查路径！")
    exit()

# 保留原图副本，用于最终对比
origin = img.copy()
# 统一缩放，加快运算（放大处理，更容易抓边缘）
height, width = img.shape[:2]
scale = 0.8
img = cv2.resize(img, (int(width*scale), int(height*scale)))
origin_resize = img.copy()

# ---------------------- 2. 灰度化 + 增强对比度 ----------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 用自适应阈值增强文档边缘
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# ---------------------- 3. 高斯滤波（去噪） ----------------------
blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

# ---------------------- 4. Canny 边缘检测（调低阈值） ----------------------
canny = cv2.Canny(blur, 30, 100)
# 膨胀边缘，让文档轮廓更连续
kernel = np.ones((3,3), np.uint8)
canny = cv2.dilate(canny, kernel, iterations=1)

# ---------------------- 5. 查找轮廓，筛选文档外轮廓 ----------------------
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 按轮廓面积从大到小排序
contours = sorted(contours, key=cv2.contourArea, reverse=True)
doc_contour = None

for cnt in contours:
    # 轮廓近似，多边形拟合
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # 放宽条件：先找顶点数为4的轮廓
    if len(approx) == 4:
        doc_contour = approx
        break

# 如果没找到四边形，就用面积最大的轮廓强行拟合
if doc_contour is None and len(contours) > 0:
    cnt = contours[0]
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    doc_contour = approx

if doc_contour is None or len(doc_contour) != 4:
    print("未检测到文档四边形轮廓！")
    exit()

# 绘制检测到的文档轮廓
cv2.drawContours(img, [doc_contour], -1, (0, 255, 0), 3)

# ---------------------- 6. 坐标排序 + 透视变换矫正 ----------------------
# 整理四个顶点坐标
pts = doc_contour.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

# 坐标排序：左上、右上、右下、左下
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]   # 左上
rect[2] = pts[np.argmax(s)]   # 右下

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]# 右上
rect[3] = pts[np.argmax(diff)]# 左下

# 计算新图像宽高
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# 目标矫正后四个角坐标
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

# 计算透视变换矩阵 & 执行变换
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(origin_resize, M, (maxWidth, maxHeight))

# 矫正后再做二值化，让文档更清晰
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
_, warp_binary = cv2.threshold(warp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------------------- 7. 显示所有阶段结果 ----------------------
cv2.imshow("1_Original", origin_resize)
cv2.imshow("2_Gray", gray)
cv2.imshow("3_Gaussian_Blur", blur)
cv2.imshow("4_Canny_Edge", canny)
cv2.imshow("5_Contour", img)
cv2.imshow("6_Warp_Result", warp)
cv2.imshow("7_Warp_Binary", warp_binary)

# ---------------------- 8. 保存最终矫正图片 ----------------------
cv2.imwrite("images/doc_result.jpg", warp)
cv2.imwrite("images/doc_binary.jpg", warp_binary)
print("矫正后的文档已保存至 images/doc_result.jpg 和 images/doc_binary.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()