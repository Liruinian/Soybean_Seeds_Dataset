import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d
import os


class ColmapVisualizer:
    def __init__(self, colmap_path="CC/"):
        # 初始化路径和参数
        self.colmap_path = colmap_path
        self.depth_dir = os.path.join(colmap_path, "depth_images")
        self.image_dir = os.path.join(colmap_path, "images")
        self.selected_image_name = "frame0024.png"

        # 初始化可视化元素
        self.fig = None
        self.ax = None
        self.vis = None
        self.click_circle = None
        self.click_point = None

        # 加载COLMAP重建数据
        self._load_colmap_reconstruction()
        self._setup_camera_parameters()
        self._load_and_preprocess_images()

        # 建立深度-距离关系模型
        self.depth_model_params = self._build_depth_distance_model()

    def _load_colmap_reconstruction(self):
        """加载COLMAP重建结果并建立索引"""
        recon_path = os.path.join(self.colmap_path, "reconstruction/0")
        self.reconstruction = pycolmap.Reconstruction(recon_path)
        self.image_name_to_id = {
            img.name: img_id for img_id, img in self.reconstruction.images.items()
        }

    def _setup_camera_parameters(self):
        """解析相机参数并创建去畸变映射"""
        image_id = self.image_name_to_id[self.selected_image_name]
        self.image = self.reconstruction.images[image_id]
        self.camera = self.reconstruction.cameras[self.image.camera_id]

        # 解析相机模型参数
        model_handlers = {
            pycolmap.CameraModelId.SIMPLE_PINHOLE: self._handle_simple_pinhole,
            pycolmap.CameraModelId.SIMPLE_RADIAL: self._handle_simple_radial,
            pycolmap.CameraModelId.RADIAL: self._handle_radial,
        }

        handler = model_handlers.get(self.camera.model)
        if not handler:
            raise NotImplementedError(f"不支持的相机模型: {self.camera.model.name}")

        self.K, self.dist_coeffs = handler()

    def _handle_simple_pinhole(self):
        params = self.camera.params
        f, cx, cy = params
        return (
            np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]),
            np.zeros(4)
        )

    def _handle_simple_radial(self):
        params = self.camera.params
        f, cx, cy, k = params
        return (
            np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]),
            np.array([k, 0, 0, 0])
        )

    def _handle_radial(self):
        params = self.camera.params
        f, cx, cy, k1, k2 = params
        return (
            np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]),
            np.array([k1, k2, 0, 0])
        )

    def _load_and_preprocess_images(self):
        """加载并预处理图像和深度图"""
        # 加载原始图像并去畸变
        img_path = os.path.join(self.image_dir, self.selected_image_name)
        self.undistorted_img = cv2.undistort(
            cv2.imread(img_path), self.K, self.dist_coeffs
        )

        # 加载并处理深度图
        self.depth_map = self._load_depth_map()
        if self.depth_map is not None:
            self._create_depth_overlay()

    def _load_depth_map(self):
        """加载并预处理深度图"""
        depth_path = os.path.join(self.depth_dir, self.selected_image_name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"警告: 深度图未找到于 {depth_path}")
            return None

        # 统一深度图格式
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = cv2.resize(depth, (self.undistorted_img.shape[1], self.undistorted_img.shape[0]))
        return depth.astype(np.uint16)

    def _create_depth_overlay(self):
        """创建深度图叠加可视化"""
        depth_normalized = cv2.normalize(
            self.depth_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        # 创建带透明度的深度图
        alpha = 0.5
        overlay = self.undistorted_img.copy()
        overlay[depth_normalized > 100] = [255, 0, 0]  # 深度大于100的区域标红
        self.undistorted_img = cv2.addWeighted(
            self.undistorted_img, 1-alpha, overlay, alpha, 0
        )

    def _build_depth_distance_model(self):
        """建立深度灰度值与3D距离的回归模型"""
        # 准备有效数据点
        valid_points = []
        for p in self.image.points2D:
            if p.point3D_id not in self.reconstruction.points3D:
                continue
            
            x, y = p.xy
            if not (0 <= x < self.depth_map.shape[1] and 0 <= y < self.depth_map.shape[0]):
                continue
            
            depth_val = self.depth_map[int(y), int(x)]
            point3d = self.reconstruction.points3D[p.point3D_id]
            distance = np.linalg.norm(self.image.cam_from_world.translation - point3d.xyz)
            valid_points.append((depth_val, distance))

        # 进行指数回归
        from scipy.optimize import curve_fit
        X, y = zip(*valid_points)
        try:
            popt, _ = curve_fit(
                lambda x, a, b: a * np.exp(b * x),
                X, y, p0=[1, 0.01]
            )
            return popt
        except RuntimeError:
            print("深度-距离模型建立失败")
            return None

    def setup_2d_visualization(self):
        """初始化2D可视化界面"""
        self.fig = plt.figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(cv2.cvtColor(self.undistorted_img, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')

        # 绘制3D关联点
        self._plot_3d_points()
        self._setup_interactive_elements()
        
    def _plot_3d_points(self):
        """绘制带深度颜色编码的3D关联点"""
        valid_points = []
        colors = []
        for p in self.image.points2D:
            if p.point3D_id not in self.reconstruction.points3D:
                continue
            
            x, y = p.xy
            valid_points.append((x, y))
            colors.append('b' if self.depth_map[int(y), int(x)] >= 100 else 'r')
        
        self.ax.scatter(
            [p[0] for p in valid_points],
            [p[1] for p in valid_points],
            c=colors, s=5, alpha=0.6
        )

    def _setup_interactive_elements(self):
        """设置交互式元素"""
        self.click_circle = plt.Circle((0,0), 20, color='y', fill=False, visible=False)
        self.ax.add_patch(self.click_circle)
        self.click_point = self.ax.scatter([], [], c='yellow', s=50, visible=False)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_3d_visualization(self):
        """初始化3D点云可视化"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1280, height=720)

        # 添加点云
        points = [p.xyz for p in self.reconstruction.points3D.values()]
        colors = [p.color/255 for p in self.reconstruction.points3D.values()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(pcd)

        # 添加相机坐标系
        self._add_camera_coordinate()

    def _add_camera_coordinate(self):
        """添加当前相机的坐标系到3D视图"""
        transform = np.eye(4)
        transform[:3, :3] = self.image.cam_from_world.rotation.matrix()
        transform[:3, 3] = self.image.cam_from_world.translation
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_frame.transform(transform)
        self.vis.add_geometry(coord_frame)

    def on_click(self, event):
        """处理2D图像的点击事件"""
        if event.xdata is None:
            return

        # 更新点击标记
        self._update_click_marker(event.xdata, event.ydata)

        # 查找邻近点
        point_info = self._find_nearest_points(event.xdata, event.ydata)
        if point_info:
            self._highlight_3d_point(*point_info)

        # 显示深度估计信息
        self._show_depth_estimation(event.xdata, event.ydata)

    def _update_click_marker(self, x, y):
        """更新点击位置的可视化标记"""
        self.click_circle.center = (x, y)
        self.click_circle.set_visible(True)
        self.click_point.set_offsets([[x, y]])
        self.click_point.set_visible(True)
        self.fig.canvas.draw_idle()

    def _find_nearest_points(self, x, y):
        """查找点击位置附近的3D点"""
        # 建立有效点的KD树
        valid_points = [
            (p.xy[0], p.xy[1]) 
            for p in self.image.points2D 
            if p.point3D_id in self.reconstruction.points3D
        ]
        kdtree = KDTree(valid_points)
        indices = kdtree.query_ball_point((x, y), r=20)

        if not indices:
            print("未找到邻近的3D点")
            return None

        point3d_id = self.image.points2D[indices[0]].point3D_id
        point3d = self.reconstruction.points3D[point3d_id]
        return point3d, point3d_id

    def _highlight_3d_point(self, point3d, point_id):
        """在3D视图中高亮显示指定点"""
        # 创建高亮箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02, 
            cone_radius=0.03,
            cylinder_height=0.2,
            cone_height=0.05
        )
        arrow.paint_uniform_color([0, 1, 0])
        arrow.translate(point3d.xyz)
        
        # 更新视图
        self.vis.add_geometry(arrow)
        self.vis.poll_events()
        self._print_point_info(point3d, point_id)

    def _print_point_info(self, point3d, point_id):
        """打印3D点详细信息"""
        print(f"\n3D点ID: {point_id}")
        print(f"坐标: {point3d.xyz}")
        print(f"颜色: {point3d.color}")
        print(f"跟踪次数: {len(point3d.track)}")

    def _show_depth_estimation(self, x, y):
        """显示深度估计信息"""
        if self.depth_map is None:
            return

        depth_val = self.depth_map[int(y), int(x)]
        print(f"\n点击位置深度值: {depth_val}")
        
        if self.depth_model_params:
            a, b = self.depth_model_params
            estimated_dist = a * np.exp(b * depth_val)
            print(f"估计距离: {estimated_dist:.2f} 米")

    def run(self):
        """运行可视化"""
        self.setup_2d_visualization()
        self.setup_3d_visualization()
        plt.show(block=False)
        self.vis.run()


if __name__ == "__main__":
    visualizer = ColmapVisualizer()
    visualizer.run()