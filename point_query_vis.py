import cv2
import numpy as np
import pycolmap
import matplotlib.pyplot as plt
import open3d as o3d
import os
from scipy.spatial import KDTree

# 配置路径
colmap_path = "CC/"
reconstruction = pycolmap.Reconstruction(colmap_path+"reconstruction/0")
selected_image_name = "frame0024.png"
seed_3d_dir = os.path.join(colmap_path, "3d_seed_pos")
seed_3d_path = os.path.join(seed_3d_dir, f"{selected_image_name.replace('.png', '')}.txt")



# 初始化可视化窗口
fig, ax = plt.subplots(figsize=(12, 8))
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Point Cloud', width=1280, height=720)

def setup_visualization():
    """初始化可视化环境"""
    # 加载COLMAP数据
    image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}
    image_id = image_name_to_id[selected_image_name]
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]

    # 创建3D点云
    pcd = o3d.geometry.PointCloud()
    all_points = [p.xyz for p in reconstruction.points3D.values()]
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(
        [p.color/255 for p in reconstruction.points3D.values()])
    vis.add_geometry(pcd)

    # 添加相机位置
    add_camera_mesh(image)

    # 加载并显示2D图像
    show_2d_image(image, camera)

    # 加载3D种子点
    load_3d_seeds()

def add_camera_mesh(image):
    """添加相机坐标系到3D视图"""
    pose = image.cam_from_world
    R = pose.rotation.matrix()
    t = pose.translation
    
    camera_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    transform = np.eye(4)
    transform[:3, :3] = R @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    transform[:3, 3] = -R.T @ t
    camera_mesh.transform(transform)
    vis.add_geometry(camera_mesh)

def show_2d_image(image, camera):
    """显示2D图像及特征点"""
    img = cv2.imread(f"{colmap_path}/images/{selected_image_name}")

    # undistorted = cv2.undistort(img, parse_camera_matrix(camera), parse_distortion(camera))
    
    # 显示图像
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # 绘制特征点
    points2D = image.points2D
    xy = np.array([(p.xy[0], p.xy[1]) for p in points2D])
    point3D_ids = [p.point3D_id for p in points2D]
    kdtree = KDTree(xy)

    # 在绘制图像后添加散点绘制
    valid_points = [(p.xy[0], p.xy[1]) for p in points2D if p.point3D_id in reconstruction.points3D]
    xy_valid = np.array(valid_points)
    point3D_ids_valid = [p.point3D_id for p in points2D if p.point3D_id in reconstruction.points3D]
    ax.scatter(*zip(*points2D), s=5, alpha=0.6, c='blue')

def parse_camera_matrix(camera):
    """解析相机内参矩阵"""
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
    elif camera.model == pycolmap.CameraModelId.PINHOLE:
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        dist_coeffs = np.zeros(4)
    else:
        print(f"不支持的相机模型: {camera.model.name}")  # 使用.name获取枚举名称
        raise NotImplementedError(f"不支持的相机模型: {camera.model.name}")


def parse_distortion(camera):
    """解析畸变参数"""
    if camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
        return np.array([camera.params[3], 0, 0, 0])
    elif camera.model == pycolmap.CameraModelId.RADIAL:
        return np.array([camera.params[3], camera.params[4], 0, 0])
    return np.zeros(4)

def load_3d_seeds():
    """加载并显示3D种子点"""
    if not os.path.exists(seed_3d_path):
        print(f"未找到种子点文件: {seed_3d_path}")
        return

    seeds = np.loadtxt(seed_3d_path)
    if seeds.size == 0:
        return

    # 创建种子点标记
    for seed in seeds:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.paint_uniform_color([1, 0, 0])  # 红色标记
        sphere.translate(seed)
        vis.add_geometry(sphere)

    # 更新渲染
    vis.poll_events()
    vis.update_renderer()

def start_visualization():
    """启动可视化循环"""
    # 启动Matplotlib
    plt.ion()
    plt.show(block=False)
    
    # 启动Open3D渲染循环
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    
    # 清理资源
    vis.destroy_window()
    plt.close()

if __name__ == "__main__":
    setup_visualization()
    start_visualization()