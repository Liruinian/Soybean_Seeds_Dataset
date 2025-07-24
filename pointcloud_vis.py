import open3d as o3d
import numpy as np
import struct

def read_colmap_bin_points(file_path):
    """
    读取COLMAP二进制格式的点云文件 (points3D.bin)
    返回open3d.geometry.PointCloud对象
    """
    points = []
    colors = []
    
    with open(file_path, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        
        for _ in range(num_points):
            # 读取固定长度部分（56字节，包含字节对齐填充）
            fixed_data = fid.read(8 + 3*8 + 3 + 5 + 8 + 8)
            if len(fixed_data) != 56:
                break
                
            # 解析字段（Q=uint64, 3d=xyz坐标, 3B=RGB颜色, 5x=填充, d=误差, Q=轨迹长度）
            (point_id, 
             x, y, z,
             r, g, b,
             error,
             track_length) = struct.unpack("Q3d3B5xdQ", fixed_data)
            
            # 修改这部分读取方式
            # 改为逐个读取track元素（每个8字节）
            for _ in range(track_length):
                fid.read(8)  # 每个track元素包含两个uint32（图像ID和特征点ID）
            
            # 或者一次性读取全部track数据（更高效）
            # fid.read(track_length * 8)
            
            points.append([x, y, z])
            colors.append([r/255.0, g/255.0, b/255.0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def visualize_point_cloud(file_path):
    pcd = read_colmap_bin_points(file_path)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加几何体
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0       # 点的大小
    render_opt.background_color = np.array([0.1, 0.1, 0.1])  # 背景颜色
    
    # 设置初始视角控制方式
    view_ctl = vis.get_view_control()
    view_ctl.set_constant_z_far(1000)  # 设置最大可视距离
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 修改为你的实际文件路径
    point_cloud_path = "CC/reconstruction/0/points3D.bin"
    visualize_point_cloud(point_cloud_path)
