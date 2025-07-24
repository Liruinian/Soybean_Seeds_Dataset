import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_density_map(points, height, width, sigma=4):
    """生成密度图"""
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            density[int(y), int(x)] = 1
    return gaussian_filter(density, sigma=sigma)

# 示例路径（根据实际数据修改）
image_path = "CC/images/frame0009.png"
label_path = "CC/seed_pos/frame0009.txt"

# 读取图像和标签
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
label_points = np.loadtxt(label_path)

# 生成密度图
density_map = generate_density_map(label_points, img.shape[0], img.shape[1], sigma=8)

# 创建可视化布局
plt.figure()

# # 显示原始图像和种子点
# plt.subplot(1, 3, 1)
# plt.imshow(img)
# plt.scatter(label_points[:, 0], label_points[:, 1], c='red', s=20, 
#            edgecolors='white', linewidths=0.5, alpha=0.8)
# plt.title("Original Image with Seed Points")
# plt.axis('off')

# # 显示密度图热力图
# plt.subplot(1, 3, 2)
# heatmap = plt.imshow(density_map, cmap='jet', alpha=0.7)
# plt.colorbar(heatmap, fraction=0.046, pad=0.04)
# plt.title("Density Heatmap")
# plt.axis('off')

# # 显示叠加效果
# plt.subplot(1, 3, 3)
plt.imshow(img)
overlay = plt.imshow(density_map, cmap='jet', alpha=0.5)
# plt.colorbar(overlay, fraction=0.046, pad=0.04)
# plt.title("Image with Density Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()