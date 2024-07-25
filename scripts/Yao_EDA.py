import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linprog


# 指定影像資料夾的路徑
current_path = os.getcwd()
image_folder_path = r"data\casting_data\train\def_front"
image_folder_path = os.path.join(current_path, image_folder_path)
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# 指定影像資料夾的路徑
current_path = os.getcwd()
defect_image_path = r"data\casting_data\train\def_front"
defect_path = os.path.join(current_path, defect_image_path)

normal_image_path = r"data\casting_data\train\ok_front"
normal_path = os.path.join(current_path, normal_image_path)

# 讀取文件列表
defect_image_files = [f for f in os.listdir(defect_path) if os.path.isfile(os.path.join(defect_path, f))]
normal_image_files = [f for f in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, f))]

# 初始化列表存儲影像數據
defect_images = []
normal_images = []

# 加載影像並存儲
for file in defect_image_files:
    image_path = os.path.join(defect_path, file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取影像
    if image is not None:
        defect_images.append(image)

for file in normal_image_files:
    image_path = os.path.join(normal_path, file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取影像
    if image is not None:
        normal_images.append(image)


# 檢查影像的尺寸和解析度
image_shapes = [img.shape for img in defect_images]
print("影像尺寸:", image_shapes)
######################
#　EMD
# 計算灰度直方圖
def compute_histogram(image, bins=256):
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 計算平均灰度直方圖
def compute_average_histogram(images, bins=256):
    avg_hist = np.zeros(bins)
    for image in images:
        hist = compute_histogram(image, bins)
        avg_hist += hist
    avg_hist /= len(images)
    return avg_hist

# 計算正常影像和缺陷影像的平均灰度直方圖
avg_hist_normal = compute_average_histogram(normal_images)
avg_hist_defect = compute_average_histogram(defect_images)

# 使用SciPy計算EMD (一維的Wasserstein distance)
def compute_emd(hist1, hist2):
    emd_value = wasserstein_distance(hist1, hist2)
    return emd_value

# 計算正常影像和缺陷影像之間的EMD，作為基準
emd_value_normal_defect = compute_emd(avg_hist_normal, avg_hist_defect)
print(f"EMD between normal and defect images: {emd_value_normal_defect}")

#########################################################################



# 計算灰度直方圖
def compute_histogram(image, bins=256):
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 計算兩個直方圖之間的歐幾里得距離矩陣
def compute_distance_matrix(hist1, hist2):
    d = np.zeros((len(hist1), len(hist2)))
    for i in range(len(hist1)):
        for j in range(len(hist2)):
            d[i, j] = abs(hist1[i] - hist2[j])
    return d

# 使用線性規劃求解EMD
def compute_emd(hist1, hist2):
    d = compute_distance_matrix(hist1, hist2)
    num_bins = len(hist1)

    # 建立線性規劃問題
    c = d.flatten()
    A_eq = np.zeros((2 * num_bins, num_bins**2))
    for i in range(num_bins):
        A_eq[i, i*num_bins:(i+1)*num_bins] = 1
        A_eq[num_bins+i, i::num_bins] = 1
    b_eq = np.concatenate([hist1, hist2])
    bounds = [(0, None)] * num_bins**2

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    emd = result.fun
    return emd

# 設置路徑
# defect_image_path = 'path_to_defect_images'
# normal_image_path = 'path_to_normal_images'

defect_image_files = [f for f in os.listdir(defect_path) if os.path.isfile(os.path.join(defect_path, f))]
normal_image_files = [f for f in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, f))]

# 計算正常影像和缺陷影像的平均灰度直方圖
def compute_average_histogram(images, bins=256):
    avg_hist = np.zeros(bins)
    for image in images:
        hist = compute_histogram(image, bins)
        avg_hist += hist
    avg_hist /= len(images)
    return avg_hist

avg_hist_normal = compute_average_histogram(normal_images)
avg_hist_defect = compute_average_histogram(defect_images)

# 計算正常影像和缺陷影像之間的EMD
emd_value_normal_defect = compute_emd(avg_hist_normal, avg_hist_defect)
print(f"EMD between normal and defect images: {emd_value_normal_defect}")





###########################################################################
# EDA

# 可視化影像
plt.figure(figsize=(10, 10))
for i, image in enumerate(defect_image_files):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Image {i + 1}')
    plt.axis('off')
plt.show()

# 計算影像的基本統計量
for i, image in enumerate(defect_image_files):
    print(f'Image {i + 1} - Mean: {np.mean(image)}, Std: {np.std(image)}, Min: {np.min(image)}, Max: {np.max(image)}')

# 可視化影像的直方圖
plt.figure(figsize=(10, 5))
for i, image in enumerate(defect_image_files):
    plt.subplot(2, 4, i + 1)
    plt.hist(image.ravel(), bins=256, color='black', alpha=0.7)
    plt.title(f'Image {i + 1} Histogram')
plt.tight_layout()
plt.show()