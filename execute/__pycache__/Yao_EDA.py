import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 指定影像資料夾的路徑
current_path = os.getcwd()
image_folder_path = r"data\casting_data\casting_data\train\def_front"
image_folder_path = os.path.join(current_path, image_folder_path)
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# 初始化列表存儲影像數據
images = []

# 加載影像並存儲
for file in image_files:
    image_path = os.path.join(image_folder_path, file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取影像
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取彩色影像
    if image is not None:
        images.append(image)

# 檢查影像的尺寸和解析度
image_shapes = [img.shape for img in images]
print("影像尺寸:", image_shapes)



# EDA

# 可視化影像
plt.figure(figsize=(10, 10))
for i, image in enumerate(images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Image {i + 1}')
    plt.axis('off')
plt.show()

# 計算影像的基本統計量
for i, image in enumerate(images):
    print(f'Image {i + 1} - Mean: {np.mean(image)}, Std: {np.std(image)}, Min: {np.min(image)}, Max: {np.max(image)}')

# 可視化影像的直方圖
plt.figure(figsize=(10, 5))
for i, image in enumerate(images):
    plt.subplot(2, 4, i + 1)
    plt.hist(image.ravel(), bins=256, color='black', alpha=0.7)
    plt.title(f'Image {i + 1} Histogram')
plt.tight_layout()
plt.show()