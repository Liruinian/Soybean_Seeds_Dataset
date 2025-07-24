import os
import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree

# 配置路径
colmap_path = "D:\Desktop\soybean-paper\Colmap_2d3d_Mapping\Soybean_Reconstruction_Images\soybean_099\\vggt\\"
reconstruction = pycolmap.Reconstruction(os.path.join(colmap_path, "sparse\\0"))

# 创建图像名称到ID的映射
image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}

# 创建输出目录
output_dir = os.path.join(colmap_path, "3d_keypoints")
os.makedirs(output_dir, exist_ok=True)

# 处理所有图片文件
image_dir = os.path.join(colmap_path, "images")
processed_count = 0

for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # 获取图像ID
    try:
        image_id = image_name_to_id[image_name]
    except KeyError:
        print(f"跳过 {image_name}，未找到对应重建数据")
        continue
    
    # 获取相机参数
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    
    # 解析相机参数
    params = camera.params
    if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        f, cx, cy = params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        dist_coeffs = np.zeros(4)
    elif camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
        f, cx, cy, k = params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        dist_coeffs = np.array([k, 0, 0, 0])
    elif camera.model == pycolmap.CameraModelId.RADIAL:
        f, cx, cy, k1, k2 = params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, 0, 0])
    else:
        print(f"不支持的相机模型: {camera.model.name}")
        continue
    
    # 加载种子点
    seed_path = os.path.join(colmap_path, "2d_seed_pos", image_name.replace('.png', '.txt'))
    try:
        seed_points = np.loadtxt(seed_path)
        if seed_points.size == 0:
            print(f"{image_name} 无有效种子点")
            continue
    except Exception as e:
        print(f"无法加载种子点文件: {seed_path} - {str(e)}")
        continue
    
    # 准备当前图像的3D点数据
    points2D = image.points2D
    valid_points = [(p.xy[0], p.xy[1]) for p in points2D if p.point3D_id in reconstruction.points3D]
    xy_valid = np.array(valid_points)
    point3D_ids_valid = [p.point3D_id for p in points2D if p.point3D_id in reconstruction.points3D]
    
    if len(xy_valid) == 0:
        print(f"{image_name} 无有效3D点")
        continue
    
    # 创建KDTree
    kdtree = KDTree(xy_valid)
    
    # 转换种子点坐标
    points = seed_points.reshape(-1, 1, 2).astype(np.float32)
    undistorted_seeds = cv2.undistortPoints(points, K, dist_coeffs, None, K).reshape(-1, 2)
    
    # 查询3D坐标
    results = []
    for seed in undistorted_seeds:
        indices = kdtree.query_ball_point(seed, r=5)
        if indices:
            closest_idx = indices[0]
            point3D = reconstruction.points3D[point3D_ids_valid[closest_idx]]
            # 修改这里：使用xyz属性代替单独x/y/z属性
            x, y, z = point3D.xyz
            results.append(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    # 保存结果
    output_path = os.path.join(output_dir, image_name.replace('.png', '.txt'))
    with open(output_path, 'w') as f:
        f.writelines(results)
    
    processed_count += 1
    print(f"已处理 {image_name}，找到 {len(results)} 个3D点")

print(f"\n处理完成！共处理 {processed_count} 张图片")