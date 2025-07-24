import cv2
import numpy as np
import pycolmap
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d
import os
from matplotlib.widgets import RectangleSelector

# Configuration
colmap_path = "CC/"
image_dir = os.path.join(colmap_path, "images")
output_dir = os.path.join(colmap_path, "selected_points")
os.makedirs(output_dir, exist_ok=True)

# Load COLMAP reconstruction
reconstruction = pycolmap.Reconstruction(colmap_path+"reconstruction/0")
image_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}

class InteractiveState:
    def __init__(self):
        self.current_image_idx = 0
        self.image_list = sorted([f for f in os.listdir(image_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.vis = None
        self.pcd = None
        self.highlight = None
        self.coord_frame = None
        self.current_kdtree = None
        self.current_point3D_ids = None
        self.current_image_name = ""

state = InteractiveState()

def init_3d_visualization():
    state.vis = o3d.visualization.Visualizer()
    state.vis.create_window(window_name='3D Point Cloud', width=1280, height=720)
    
    # Highlight sphere
    state.highlight = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    state.highlight.paint_uniform_color([1, 0, 1])  # Magenta
    state.vis.add_geometry(state.highlight)
    
    # Coordinate frame
    state.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    state.vis.add_geometry(state.coord_frame)
    
    # Initialize point cloud
    all_points = np.array([p.xyz for p in reconstruction.points3D.values()])
    colors = np.array([p.color / 255.0 for p in reconstruction.points3D.values()])
    
    state.pcd = o3d.geometry.PointCloud()
    state.pcd.points = o3d.utility.Vector3dVector(all_points)
    state.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    state.vis.add_geometry(state.pcd)
    
    # Set initial view
    view_ctl = state.vis.get_view_control()
    view_ctl.set_front([-0.5, -0.3, 0.8])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.1)

def update_3d_visualization(center):
    """Update 3D visualization elements"""
    # Move highlight sphere
    state.highlight.translate(-state.highlight.get_center(), relative=False)
    state.highlight.translate(center, relative=False)
    
    # Move coordinate frame
    state.coord_frame.translate(-state.coord_frame.get_center(), relative=False)
    state.coord_frame.translate(center, relative=False)
    
    # Update render
    state.vis.update_geometry(state.highlight)
    state.vis.update_geometry(state.coord_frame)
    state.vis.poll_events()
    state.vis.update_renderer()

def process_image(image_name):
    """Process single image"""
    state.current_kdtree = None
    state.current_point3D_ids = None
    state.current_image_name = image_name
    
    try:
        # Load image data
        image_id = image_name_to_id[image_name]
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        
        # Camera parameters
        params = camera.params
        if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
            f, cx, cy = params
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist_coeffs = np.zeros(4)
        elif camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
            f, cx, cy, k = params
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist_coeffs = np.array([k, 0, 0, 0])
        elif camera.model == pycolmap.CameraModelId.RADIAL:
            f, cx, cy, k1, k2 = params
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, 0, 0])
        else:
            raise NotImplementedError(f"Unsupported camera model: {camera.model.name}")
        
        # Load and undistort image
        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)
        undistorted_img = cv2.undistort(img, K, dist_coeffs)
        undistorted_img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
        
        # Prepare 2D points
        points2D = image.points2D
        valid_points = [(p.xy[0], p.xy[1]) for p in points2D if p.point3D_id in reconstruction.points3D]
        xy_valid = np.array(valid_points)
        point3D_ids_valid = [p.point3D_id for p in points2D if p.point3D_id in reconstruction.points3D]
        
        # Create KDTree
        if len(xy_valid) > 0:
            state.current_kdtree = KDTree(xy_valid)
            state.current_point3D_ids = point3D_ids_valid
        else:
            print(f"No valid 3D points found in {image_name}")
        
        return undistorted_img_rgb, xy_valid
    
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        return None, None
def on_select(eclick, erelease):
    """Rectangle selection callback"""
    if state.current_kdtree is None:
        return
    
    # 获取选区坐标并转换为图像坐标系
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    
    # 精确查询矩形区域内的点
    valid_points = state.current_kdtree.data
    in_rect_mask = (
        (valid_points[:, 0] >= x_min) & 
        (valid_points[:, 0] <= x_max) & 
        (valid_points[:, 1] >= y_min) & 
        (valid_points[:, 1] <= y_max)
    )
    indices = np.where(in_rect_mask)[0]
    
    # 收集3D点并更新可视化
    selected_points = []
    for idx in indices:
        point3D_id = state.current_point3D_ids[idx]
        point3D = reconstruction.points3D[point3D_id]
        selected_points.append(point3D.xyz)
        update_3d_visualization(point3D.xyz)
    
    # 保存结果
    if selected_points:
        output_path = os.path.join(output_dir, f"selected_{state.current_image_name.replace('.png', '.txt')}")
        np.savetxt(output_path, selected_points)
        print(f"成功保存 {len(selected_points)} 个点到 {output_path}")
    else:
        print("当前选区没有检测到3D点")



def on_key(event):
    """Keyboard event handler"""
    if event.key == 'n':
        state.current_image_idx += 1
        plt.close()
    elif event.key == 'q':
        if state.vis:
            state.vis.destroy_window()
        plt.close()

# Initialize 3D visualization
init_3d_visualization()

# Main processing loop
while state.current_image_idx < len(state.image_list):
    image_name = state.image_list[state.current_image_idx]
    
    try:
        img_rgb, xy_valid = process_image(image_name)
        if img_rgb is None:
            state.current_image_idx += 1
            continue
        
        # Create matplotlib interface
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.imshow(img_rgb)
        
        # Plot 2D points with 3D associations
        if xy_valid is not None and len(xy_valid) > 0:
            ax.scatter(xy_valid[:, 0], xy_valid[:, 1], 
                      c='r', s=15, edgecolors='white', 
                      alpha=0.7, label='3D Points')
            ax.legend()
        
        ax.set_title(f"Current Image: {image_name} (N: Next, Q: Quit)")
        ax.axis('off')
        
        # Bind events
        RectangleSelector(ax, on_select, useblit=True,
                props=dict(facecolor='yellow', edgecolor='red', alpha=0.3, linestyle='--'))
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        
    except Exception as e:
        print(f"Error displaying {image_name}: {str(e)}")
    
    # Auto-save all valid points
    output_path = os.path.join(output_dir, image_name.replace('.png', '.txt'))
    if state.current_point3D_ids:
        valid_3d_points = [reconstruction.points3D[pid].xyz for pid in state.current_point3D_ids]
        np.savetxt(output_path, valid_3d_points)
    
    state.current_image_idx += 1

# Cleanup
if state.vis:
    state.vis.destroy_window()
print("Processing completed!")