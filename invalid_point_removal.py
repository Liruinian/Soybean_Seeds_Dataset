import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d

reconstruction = pycolmap.Reconstruction("reconstruction/0")

image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}

selected_image_name = "frame0009.png" 
image_id = image_name_to_id[selected_image_name]
image = reconstruction.images[image_id]
camera = reconstruction.cameras[image.camera_id]

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
    print(f"不支持的相机模型: {camera.model.name}")  # 使用.name获取枚举名称
    raise NotImplementedError(f"不支持的相机模型: {camera.model.name}")

img_path = "./images/" + selected_image_name  # 图片路径
img = cv2.imread(img_path)
undistorted_img = cv2.undistort(img, K, dist_coeffs)
undistorted_img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

height, width = undistorted_img.shape[:2]
dpi = 100
fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(undistorted_img_rgb)
ax.axis('off')

points2D = image.points2D
xy = np.array([(p.xy[0], p.xy[1]) for p in points2D])
point3D_ids = [p.point3D_id for p in points2D]
kdtree = KDTree(xy)

# 在绘制图像后添加散点绘制
valid_points = [(p.xy[0], p.xy[1]) for p in points2D if p.point3D_id in reconstruction.points3D]
xy_valid = np.array(valid_points)
point3D_ids_valid = [p.point3D_id for p in points2D if p.point3D_id in reconstruction.points3D]

# 只对有3D关联的点建立KD树
kdtree = KDTree(xy_valid)
ax.scatter(xy_valid[:, 0], xy_valid[:, 1], c='r', s=5, alpha=0.6, label='3D关联点')

# 修改点击反馈图形元素的创建方式
click_circle = plt.Circle((0,0), 20, color='y', fill=False, lw=2, visible=False)
ax.add_patch(click_circle)
click_point = ax.scatter([], [], c='yellow', s=50, edgecolors='black', visible=False)

# 提取所有3D点数据
all_points = np.array([p.xyz for p in reconstruction.points3D.values()])
colors = np.array([p.color / 255.0 for p in reconstruction.points3D.values()])  # 使用COLMAP存储的颜色

# === 新增形态检测部分 ===
# 创建Open3D点云对象
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(all_points)
pcd_raw.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 1. 地面检测 (使用RANSAC平面分割)
plane_model, inliers_ground = pcd_raw.segment_plane(
    distance_threshold=0.05,  # 根据实际地面平整度调整
    ransac_n=3,
    num_iterations=1000
)
pcd_ground = pcd_raw.select_by_index(inliers_ground)
pcd_nonground = pcd_raw.select_by_index(inliers_ground, invert=True)

# 2. 离群点去除 (统计滤波)
cl, ind_stats = pcd_nonground.remove_statistical_outlier(
    nb_neighbors=50,    # 考虑邻居数量
    std_ratio=0.5      # 标准差比例
)
pcd_clean = pcd_nonground.select_by_index(ind_stats)

# 3. 高度特征过滤 (假设地面是Z轴最低平面)
if len(pcd_ground.points) > 0:
    ground_z = np.asarray(pcd_ground.points)[:,2].mean()
    # 获取所有非地面点的高度
    points = np.asarray(pcd_clean.points)
    z_values = points[:,2] - ground_z
    
    # 根据高度筛选 (假设植株高度在0.1-1.5米之间)
    valid_height = (z_values > 0.1) & (z_values < 1.5)
    pcd_plants = pcd_clean.select_by_index(np.where(valid_height)[0])
    
    # 合并有效点云
    final_points = np.asarray(pcd_plants.points)
    final_colors = np.asarray(pcd_plants.colors)
    
    # 获取非地面点的原始索引
    nonground_indices = [i for i in range(len(all_points)) if i not in inliers_ground]

    # 获取离群点在原始点云中的索引
    outlier_mask = np.ones(len(nonground_indices), dtype=bool)
    outlier_mask[ind_stats] = False
    outlier_indices = np.array(nonground_indices)[outlier_mask].tolist()

    # 获取高度异常点在原始点云中的索引 (需要先获取经过统计滤波后的有效索引)
    if len(pcd_ground.points) > 0:
        # 获取统计滤波后的有效点索引
        clean_indices = np.array(nonground_indices)[ind_stats]
        # 高度异常的索引（相对于clean_indices）
        height_invalid = np.where(~valid_height)[0]
        # 转换为原始索引
        height_invalid_indices = clean_indices[height_invalid].tolist()
    else:
        height_invalid_indices = []

    invalid_indices = list(inliers_ground) + outlier_indices + height_invalid_indices
else:
    final_points = all_points
    final_colors = colors[:, :3]

# 创建最终显示的点云（有效点保持原色，无效点设为红色）
valid_pcd = o3d.geometry.PointCloud()
valid_pcd.points = o3d.utility.Vector3dVector(final_points)
valid_pcd.colors = o3d.utility.Vector3dVector(final_colors)

invalid_pcd = o3d.geometry.PointCloud()
invalid_points = np.asarray(pcd_raw.select_by_index(invalid_indices).points)
invalid_pcd.points = o3d.utility.Vector3dVector(invalid_points)
invalid_pcd.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(invalid_points),1))) # 红色

# === 修改可视化部分 ===
# 初始化可视化窗口时添加两个点云
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Filtered Point Cloud', width=1280, height=720)
vis.add_geometry(valid_pcd)    # 有效点保持原色
vis.add_geometry(invalid_pcd)  # 无效点显示为红色

# 创建高亮点
highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
highlight_sphere.paint_uniform_color([1, 0, 1])  # 改为品红色
highlight_sphere.compute_vertex_normals()

# 添加箭头标记
arrow = o3d.geometry.TriangleMesh.create_arrow(
    cylinder_radius=0.05, 
    cone_radius=0.08,
    cylinder_height=0.5,
    cone_height=0.1
)
arrow.paint_uniform_color([0, 1, 0])  # 绿色箭头

# 添加坐标轴和文本标签
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# 配置视图参数
view_ctl = vis.get_view_control()
view_ctl.set_front([-0.5, -0.3, 0.8])
view_ctl.set_up([0, -1, 0])
view_ctl.set_zoom(0.1)

# 更新3D高亮显示的函数
def update_3d_highlight(point3D, point_id):
    """更新Open3D高亮显示"""
    center = point3D.xyz
    # 移动高亮元素到目标位置
    highlight_sphere.translate(center - highlight_sphere.get_center(), relative=False)
    coord_frame.translate(center - coord_frame.get_center(), relative=False)
    
    # 定位箭头到点上方
    arrow.translate(center + [0, 0.5, 0] - arrow.get_center(), relative=False)
    
    # 在控制台打印详细信息
    print(f"\n=== 3D点详细信息 ===")
    print(f"ID: {point_id}")
    print(f"坐标: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}")
    print(f"颜色: {point3D.color}")
    print(f"跟踪次数: {len(point3D.track.elements)}")
    
    # 调整视角
    view_ctl.set_lookat(center)
    view_ctl.camera_local_translate(forward=0, right=0, up=0)  # 明确三个方向的移动量
    
    # 更新几何体
    vis.update_geometry(highlight_sphere)
    vis.update_geometry(coord_frame)
    vis.update_geometry(arrow)
    vis.poll_events()
    vis.update_renderer()

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    
    # 清除上次点击的标记
    click_circle.set_visible(False)
    click_point.set_visible(False)
    
    # 查询半径20像素内的所有点
    click = (event.xdata, event.ydata)
    indices = kdtree.query_ball_point(click, r=20)
    
    if indices:
        # 更新圆形标记位置
        click_circle.center = click
        click_circle.set_visible(True)
        
        # 更新点击点标记
        click_point.set_offsets([click])
        click_point.set_visible(True)
        
        # 绘制匹配范围
        closest_idx = indices[0]
        closest_3D_id = point3D_ids_valid[closest_idx]
        point = xy_valid[closest_idx]
        
        # 打印3D点信息
        point3D = reconstruction.points3D[closest_3D_id]
        print(f"最近3D点 ID: {closest_3D_id}")
        print(f"坐标: {point3D.xyz}")
        print(f"共找到 {len(indices)} 个邻近点")
        
        # 更新Open3D可视化时传递点ID
        update_3d_highlight(point3D, closest_3D_id)
    
    fig.canvas.draw_idle()

# 绑定事件并显示
fig.canvas.mpl_connect('button_press_event', onclick)

# 设置3D视图参数
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

plt.tight_layout()
plt.show()  # 先显示Matplotlib窗口

# 保持Open3D窗口运行
vis.run()
vis.destroy_window()