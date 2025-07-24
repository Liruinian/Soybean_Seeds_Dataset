import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d
import os

colmap_path = "VID_20240909_143428/"
reconstruction = pycolmap.Reconstruction(colmap_path+"reconstruction/0")
depth_dir = colmap_path + "depth_images"
seed_3d_dir = os.path.join(colmap_path, "3d_seed_pos")
selected_image_name = "frame_00005.png" 
os.makedirs(seed_3d_dir, exist_ok=True)
seed_3d_path = os.path.join(seed_3d_dir, f"{selected_image_name.replace('.png', '')}.txt")

# reconstruction = pycolmap.Reconstruction(colmap_path+"sparse_dense_merge")
image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}


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
elif camera.model == pycolmap.CameraModelId.PINHOLE:
    fx, fy, cx, cy = params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    dist_coeffs = np.zeros(4)
else:
    print(f"不支持的相机模型: {camera.model.name}")  # 使用.name获取枚举名称
    raise NotImplementedError(f"不支持的相机模型: {camera.model.name}")


def project_pixel_to_ray(x, y):
    """将像素坐标转换为归一化射线方向（相机坐标系）"""
    uv = np.array([x, y, 1.0])
    ray_dir = np.linalg.inv(K) @ uv
    return ray_dir / np.linalg.norm(ray_dir)


img_path = colmap_path+"/images/" + selected_image_name  # 图片路径
img = cv2.imread(img_path)
undistorted_img = cv2.undistort(img, K, dist_coeffs)

# Load depth image and create overlay
depth_path = os.path.join(depth_dir, selected_image_name)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if depth is not None:
    # Convert to single channel if needed
    if depth.ndim == 3:
        depth = depth[:,:,0]  # Take first channel if RGB image
    
    # Resize depth to match image size
    depth = cv2.resize(depth, (undistorted_img.shape[1], undistorted_img.shape[0]))
    
    # Normalize depth to 0-255 if needed
    if depth.dtype == np.uint16:
        depth = (depth / 256).astype(np.uint8)
    
    # Convert grayscale to BGR and create overlay
    # depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    # undistorted_img = cv2.addWeighted(undistorted_img, 0.5, depth_bgr, 0.5, 0)
    # 归一化深度图到0-255范围
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    depth_bgra = np.zeros((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)
    depth_bgra[..., 3] = 255 - depth_normalized  # alpha通道
    
    # 将原图转换为BGRA格式
    undistorted_bgra = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2BGRA)
    
    # 修改混合方式为正确的alpha混合
    # 分离alpha通道
    alpha = depth_bgra[..., 3] / 255.0
    alpha = alpha[..., np.newaxis]
    
    # 进行alpha混合: output = foreground * alpha + background * (1 - alpha)
    blended = (depth_bgra[..., :3] * alpha) + (undistorted_bgra[..., :3] * (1 - alpha))
    undistorted_img = blended.astype(np.uint8)


else:
    print(f"Warning: Depth image not found at {depth_path}")

height, width = undistorted_img.shape[:2]
dpi = 100
fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])

undistorted_img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    
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

all_points = np.array([p.xyz for p in reconstruction.points3D.values()])
point3d_kdtree = KDTree(all_points)

# 生成颜色数组：depth >= 100的点用蓝色，其他用红色
colors_2d = []
for (x, y_pt), point3D_id in zip(xy_valid, point3D_ids_valid):
    x_int = int(round(x))
    y_int = int(round(y_pt))
    if 0 <= x_int < depth.shape[1] and 0 <= y_int < depth.shape[0]:
        depth_value = depth[y_int, x_int]
        colors_2d.append('b' if depth_value >= 100 else 'r')
    else:
        colors_2d.append('r')  # 默认颜色

ax.scatter(xy_valid[:, 0], xy_valid[:, 1], c=colors_2d, s=5, alpha=0.6, label='3D关联点')

def auto_click_seed_points(seed_points):
    valid_count = 0
    print(f"\n开始过滤并处理种子点（共{len(seed_points)}个）...")
    
    for i, (x, y) in enumerate(seed_points):
        x_int = int(round(x))
        y_int = int(round(y))
        
        # 检查坐标有效性并获取深度值
        if 0 <= x_int < depth.shape[1] and 0 <= y_int < depth.shape[0]:
            depth_value = depth[y_int, x_int]
            # if depth_value <= 100:
            #     print(f"跳过第{i+1}个种子点：深度值{depth_value}")
            #     continue
        else:
            print(f"跳过第{i+1}个种子点：坐标({x_int},{y_int})超出图像范围")
            continue
        
        print(f"\n处理第{valid_count+1}个有效种子点 (原始第{i+1}个，深度{depth_value})")
        # 创建模拟点击事件
        class MockEvent:
            def __init__(self, x, y):
                self.xdata = x
                self.ydata = y
        # 执行点击处理
        onclick(MockEvent(x, y))
        valid_count += 1
        # # 更新显示并保持短暂时间
        # plt.pause(0.5)
    
    print(f"\n处理完成，共处理{valid_count}个有效种子点")

seed_path = f"{colmap_path}/seed_pos/{selected_image_name.replace('.png', '.txt')}"
try:
    seed_points = np.loadtxt(seed_path)
    if seed_points.size > 0:
        # 如果文件是二维数组（多个点）
        ax.scatter(seed_points[:, 0], seed_points[:, 1], c='g', s=40, marker='x', 
                  linewidths=2, label='种子点')
        print(f"成功加载 {len(seed_points)} 个种子点")
except Exception as e:
    print(f"无法加载种子点文件: {seed_path}")
    print(f"错误信息: {str(e)}")

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

# # 创建高亮点
# highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
# highlight_sphere.paint_uniform_color([1, 0, 1])  # 改为品红色
# highlight_sphere.compute_vertex_normals()

# 添加箭头标记
arrow = o3d.geometry.TriangleMesh.create_arrow(
    cylinder_radius=0.05, 
    cone_radius=0.08,
    cylinder_height=1,
    cone_height=0.1
)
arrow.paint_uniform_color([0, 1, 0])  # 绿色箭头

# 添加坐标轴和文本标签
# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

# 初始化可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Point Cloud', width=1280, height=720)
vis.add_geometry(pcd)
# vis.add_geometry(highlight_sphere)
# vis.add_geometry(coord_frame)
vis.add_geometry(arrow)

camera_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

# 使用新的cam_from_world属性获取相机位姿
pose = image.cam_from_world
R = pose.rotation.matrix()  # 直接获取旋转矩阵
tvec = pose.translation      # 获取平移向量

def camera_to_world(point_camera):
    """
    将相机坐标系中的3D点转换到世界坐标系
    :param point_camera: 相机坐标系中的点 (3,)
    :return: 世界坐标系中的点 (3,)
    """
    return R.T @ (point_camera - tvec)
def world_to_camera(point_world):
    """
    将世界坐标系中的3D点转换到相机坐标系
    :param point_world: 世界坐标系中的点 (3,)
    :return: 相机坐标系中的点 (3,)
    """
    # return R @ point_world + tvec
    return R @ (point_world - tvec) 
def camera_to_viscamera(R, t):
    """
    将COLMAP的相机位姿转换为可视化坐标系
    :param R: 旋转矩阵 (3x3)
    :param t: 平移向量 (3,)
    :return: (R_vis, t_vis) 转换后的位姿
    """
    R_vis = R.T
    t_vis = -R.T @ t
    return R_vis, t_vis
# 计算相机中心位置
# R, tvec = camera_to_viscamera(R,tvec)
camera_center = -np.linalg.inv(R) @ tvec
# 构建变换矩阵
transform = np.eye(4)
transform[:3, :3] = R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # 调整坐标系
transform[:3, 3] = camera_center  # 使用计算得到的相机中心

# 应用变换并添加几何体
camera_mesh.transform(transform)
vis.add_geometry(camera_mesh)



# 配置视图参数
view_ctl = vis.get_view_control()

def build_depth_distance_relationship():
    """构建深度灰度值与3D点距离的指数关系（双重过滤条件）"""
    X = []
    y = []
    
    # 遍历所有有效点
    for (x, y_pt), point3D_id in zip(xy_valid, point3D_ids_valid):
        if point3D_id not in reconstruction.points3D:
            continue
        
        # 获取3D点距离
        point3D = reconstruction.points3D[point3D_id]
        distance = np.linalg.norm(camera_center - point3D.xyz)
        
        # # 第一层过滤：距离>30米
        # if distance > 30:
        #     continue
        
        # 获取深度值
        x_int = int(round(x))
        y_int = int(round(y_pt))
        if 0 <= x_int < depth.shape[1] and 0 <= y_int < depth.shape[0]:
            depth_value = depth[y_int, x_int]
            
            # if depth_value < 100:
            #     continue  # 新增深度过滤条件
            
            X.append(float(depth_value))
            y.append(float(distance))
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    # 进行指数回归
    from scipy.optimize import curve_fit
    if len(X) > 1:
        print(f"使用{len(X)}个有效数据点进行拟合")

    # MAD离群点过滤
    if len(y) > 0:
        median_y = np.median(y)
        deviation = np.abs(y - median_y)
        mad = np.median(deviation)
        # 处理可能为零的MAD
        if mad == 0:
            mad = 1e-6  # 添加微小值避免除以零
        threshold = 3 * mad  # 使用3倍MAD作为阈值
        mask = (y >= (median_y - threshold)) & (y <= (median_y + threshold))
        X = X[mask]
        y = y[mask]
        print(f"MAD过滤: 保留{len(y)}个数据点 (移除{len(deviation)-len(y)}个离群点)")
    

    # 进行指数回归
    from scipy.optimize import curve_fit
    if len(X) > 1:
        # 定义指数函数
        def exp_func(x, a, b):
            return a * np.exp(b * x)  # 现在x是numpy数组
        
        try:
            # 进行非线性拟合，添加边界条件防止溢出
            popt, pcov = curve_fit(exp_func, X, y, p0=[1, 0.1], bounds=(-np.inf, [1e5, 0.5]))
            a, b = popt
        except RuntimeError as e:
            print(f"拟合失败: {str(e)}")
            return None, None, None
        return a, b
    else:
        return None, None, None

# 初始化时建立关系模型（修改变量名）
a, b = build_depth_distance_relationship()

def estimate_distance_from_depth(depth_value):
    """根据深度值估计距离（指数关系）"""
    if a is not None and b is not None:
        return a * np.exp(b * depth_value)
    else:
        return None
# view_ctl.set_front([-0.5, -0.3, 0.8])
# view_ctl.set_up([0, -1, 0])
# view_ctl.set_zoom(0.1)
# 更新3D高亮显示的函数
def update_3d_highlight(point3D, point_id):
    """更新Open3D高亮显示"""
    center = point3D.xyz
    
    # 重置高亮元素的位置
    # highlight_sphere.translate(-highlight_sphere.get_center(), relative=False)
    # coord_frame.translate(-coord_frame.get_center(), relative=False)
    arrow.translate(-arrow.get_center(), relative=False)
    
    # 移动高亮元素到目标位置
    # highlight_sphere.translate(center, relative=False)
    # coord_frame.translate(center, relative=False)
    
    # 定位箭头到点上方
    arrow.translate(center + [0, 0.5, 0], relative=False)
    
    # 在控制台打印详细信息
    print(f"3D点详细信息：")
    print(f"ID: {point_id}")
    print(f"坐标: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}")
    print(f"颜色: {point3D.color}")
    print(f"跟踪次数: {len(point3D.track.elements)}")
    distance = np.linalg.norm(camera_center - center)
    print(f"与相机位置的距离: {distance:.2f} 米")
    # # 调整视角
    # view_ctl.set_lookat(center)
    # view_ctl.camera_local_translate(forward=0, right=0, up=0)  # 明确三个方向的移动量
    
    # 更新几何体
    # vis.update_geometry(highlight_sphere)
    # vis.update_geometry(coord_frame)
    vis.update_geometry(arrow)
    vis.poll_events()
    vis.update_renderer()
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    print("== 点击事件 ==")
    # 清除上次点击的标记
    click_circle.set_visible(False)
    click_point.set_visible(False)
    
    # 查询半径20像素内的所有点
    click = (event.xdata, event.ydata)
    indices = kdtree.query_ball_point(click, r=20)

    x = int(round(click[0]))
    y = int(round(click[1]))
    depth_value = None
    if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
        depth_value = depth[y, x]
        print(f"深度值: {depth_value} (原始灰度值)")
        estimated_distance = estimate_distance_from_depth(depth_value)
        if estimated_distance is not None:

            # 获取相机参数
            if camera.model in [pycolmap.CameraModelId.SIMPLE_PINHOLE,
                                pycolmap.CameraModelId.SIMPLE_RADIAL,
                                pycolmap.CameraModelId.RADIAL]:
                # 解析相机参数
                if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    f, cx, cy = camera.params
                else:
                    f, cx, cy = camera.params[:3]
                
                # 反投影计算
                u = x  # 图像x坐标
                v = y  # 图像y坐标
                d = estimated_distance  # 估计的距离
                
                # 计算归一化坐标
                x_normalized = (u - cx) / f
                y_normalized = (v - cy) / f
                
                # 相机坐标系坐标
                X_cam = x_normalized * d
                Y_cam = y_normalized * d
                Z_cam = d
                
                # 转换到世界坐标系
                point_cam = np.array([X_cam, Y_cam, Z_cam])
                point_world = R.T @ (point_cam - tvec)
                
                print(f"基于深度的估计的3D坐标（世界坐标系）: X={point_world[0]:.2f}m, Y={point_world[1]:.2f}m, Z={point_world[2]:.2f}m")
                                # 在3D视图中添加临时标记
                    # 生成射线采样点（从相机中心沿视角方向延伸）
                ray_direction = point_world - camera_center
                ray_length = np.linalg.norm(ray_direction)
                sample_steps = np.linspace(0, ray_length, 100)
                
                # 修正广播维度问题
                sample_points = camera_center.reshape(3, 1) + (ray_direction / ray_length).reshape(3, 1) * sample_steps
                
                # 查询各采样点附近的点云密度
                densities = []
                for pt in sample_points.T:
                    count = len(point3d_kdtree.query_ball_point(pt, r=0.5))  # 0.5米搜索半径
                    densities.append(count)
                
                # 找到密度最大的区域
                max_density_idx = np.argmax(densities)
                selected_point = sample_points[:, max_density_idx]
                
                # 批量保存3D种子点

                # 使用追加模式写入，保留历史数据
                with open(seed_3d_path, 'a') as f:
                    np.savetxt(f, selected_point.reshape(1,3), fmt='%.6f', delimiter=' ')

                # 在密度最大的位置添加标记
                if densities[max_density_idx] > 5:  # 密度阈值设为5个点
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    highlight_sphere.paint_uniform_color([1, 0.5, 0])  # 橙色标记
                    highlight_sphere.translate(selected_point)
                    vis.add_geometry(highlight_sphere)
                    
                    # 在控制台输出密度信息
                    print(f"最大密度区域发现 {densities[max_density_idx]} 个点")
                else:
                    print("警告：未找到足够密集的点云区域")

                # 在3D视图中绘制射线（新增部分）
                line_points = [camera_center, point_world]  # 射线起点（相机位置）和终点（估计点）
                current_ray = o3d.geometry.LineSet()
                current_ray.points = o3d.utility.Vector3dVector(line_points)
                current_ray.lines = o3d.utility.Vector2iVector([[0, 1]])
                current_ray.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色射线

                print(f"射线方向（相机坐标系）: {ray_direction}")
                print(f"对应3D点（世界坐标系）: {selected_point}")
                if hasattr(vis, 'current_ray'):
                    vis.remove_geometry(vis.current_ray)
                vis.add_geometry(current_ray)
                temp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                temp_sphere.paint_uniform_color([1, 0, 0])  # 红色
                temp_sphere.translate(point_world)
                vis.add_geometry(temp_sphere)
                vis.poll_events()
                vis.update_renderer()
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
        
        # 新增相机距离计算
        point3D = reconstruction.points3D[closest_3D_id]
        
        # point = xy_valid[closest_idx]
        
        print(f"共找到 {len(indices)} 个邻近点")
        
        # 更新Open3D可视化时传递点ID
        update_3d_highlight(point3D, closest_3D_id)
    else:
        print("点击位置未找到三维对应点")
        # estimated_distance = estimate_distance_from_depth(depth_value)
        # if estimated_distance is not None:
        #     print(f"基于深度值估计的距离: {estimated_distance:.2f} 米 (R²={r_squared:.4f})")
        # else:
        #     print("无法进行距离估计：未建立回归模型")

    fig.canvas.draw_idle()

# 绑定事件并显示
fig.canvas.mpl_connect('button_press_event', onclick)

# 设置3D视图参数
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

plt.tight_layout()
plt.show(block=False)  # 先显示Matplotlib窗口
plt.pause(1) 



auto_click_seed_points(seed_points)


# 保持Open3D窗口运行
vis.run()
vis.destroy_window()
