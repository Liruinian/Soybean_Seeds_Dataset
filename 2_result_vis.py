import os
import open3d as o3d
import numpy as np
import pycolmap

def visualize_3d_points_and_cameras():
    # 配置路径
    colmap_path = "D:\Desktop\soybean-paper\Colmap_2d3d_Mapping\Soybean_Reconstruction_Images\soybean_099\colmap\\"
    reconstruction = pycolmap.Reconstruction(os.path.join(colmap_path, "sparse\\0"))
    keypoints_dir = os.path.join(colmap_path, "3d_keypoints")

    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    # 加载并合并所有3D关键点
    point_counter = {}
    file_counter = {}
    all_points = []
    
    for fname in os.listdir(keypoints_dir):
        if fname.endswith('.txt'):
            file_path = os.path.join(keypoints_dir, fname)
            points = np.loadtxt(file_path)
            if points.size == 0:
                continue
            
            # 处理单点情况
            if len(points.shape) == 1:
                points = points[np.newaxis, :]
            
            # 统计每个文件中的唯一点
            unique_in_file = set()
            for pt in points:
                # 将坐标转换为字符串用于精确匹配（保留4位小数）
                key = tuple(np.round(pt, 4))
                unique_in_file.add(key)
                
                # 全局计数
                point_counter[key] = point_counter.get(key, 0) + 1
            
            # 文件级计数
            for key in unique_in_file:
                file_counter[key] = file_counter.get(key, 0) + 1
            
            all_points.append(points)
    
    # 打印统计结果
    print("\n跨文件关键点出现统计:")
    sorted_points = sorted(file_counter.items(), 
                         key=lambda x: x[1], reverse=True)
    for (x, y, z), count in sorted_points:
        print(f"坐标 ({x:.4f}, {y:.4f}, {z:.4f}) 在 {count} 个文件中出现")
    
    if len(all_points) == 0:
        print("未找到3D关键点数据")
        return
    
    # 创建点云并设置颜色
    point_cloud = o3d.geometry.PointCloud()
    points_array = np.vstack(all_points)
    point_cloud.points = o3d.utility.Vector3dVector(points_array)

    # 生成颜色映射（从亮黄到深红）
    colors = []
    for pt in points_array:
        key = tuple(np.round(pt, 4))
        count = file_counter.get(key, 0)
        
        # 计算颜色渐变（出现次数越多越红）
        red = 0.5 + count/(max(1, max(file_counter.values())) * 2)
        green = 0.5 - count/(max(1, max(file_counter.values())) * 2)
        colors.append([red, green, 0])  # RGB格式

    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    vis.add_geometry(point_cloud)

    # 添加COLMAP相机位姿
    camera_coords = []
    for img_id in reconstruction.images:
        image = reconstruction.images[img_id]
        
        # 获取相机变换矩阵 (新API)
        cam_from_world = image.cam_from_world.matrix()
        R = cam_from_world[:3, :3]  # 旋转矩阵
        t = cam_from_world[:3, 3]   # 平移向量
        camera_center = -np.linalg.inv(R) @ t  # 计算相机中心

        # 创建坐标系
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # 应用变换
        camera_frame.rotate(R.T)  # 使用旋转矩阵的转置
        camera_frame.translate(camera_center)
        vis.add_geometry(camera_frame)

        camera_coords.append(camera_center)

    # 添加相机中心点云
    camera_points = o3d.geometry.PointCloud()
    camera_points.points = o3d.utility.Vector3dVector(np.array(camera_coords))
    camera_points.paint_uniform_color([1, 0, 0])  # 红色表示相机位置
    vis.add_geometry(camera_points)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, -1, 0.5])  # 设置初始视角
    view_ctl.set_up([0, 0, 1])        # 设置上方向

    # 添加坐标系
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(world_frame)

    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_points_and_cameras()