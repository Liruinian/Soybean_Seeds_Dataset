import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d

colmap_path = "VID_20240909_143428/"
reconstruction = pycolmap.Reconstruction(colmap_path+"reconstruction/1")

image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}
print(image_name_to_id)
selected_image_name = "frame_00413.png" 
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

img_path = colmap_path+"/images/" + selected_image_name  # 图片路径
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

# 创建Open3D点云可视化窗口
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # 使用RGB颜色，忽略alpha通道

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

# 初始化可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Point Cloud', width=1280, height=720)
vis.add_geometry(pcd)
vis.add_geometry(highlight_sphere)
vis.add_geometry(coord_frame)
vis.add_geometry(arrow)

# 配置视图参数
view_ctl = vis.get_view_control()
view_ctl.set_front([-0.5, -0.3, 0.8])
view_ctl.set_up([0, -1, 0])
view_ctl.set_zoom(0.1)
# 更新3D高亮显示的函数
def update_3d_highlight(point3D, point_id):
    """更新Open3D高亮显示"""
    center = point3D.xyz
    
    # 重置高亮元素的位置
    highlight_sphere.translate(-highlight_sphere.get_center(), relative=False)
    coord_frame.translate(-coord_frame.get_center(), relative=False)
    arrow.translate(-arrow.get_center(), relative=False)
    
    # 移动高亮元素到目标位置
    highlight_sphere.translate(center, relative=False)
    coord_frame.translate(center, relative=False)
    
    # 定位箭头到点上方
    arrow.translate(center + [0, 0.5, 0], relative=False)
    
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