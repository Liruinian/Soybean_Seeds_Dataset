import numpy as np
import struct

def read_geometric_bin(file_path):
    with open(file_path, 'rb') as f:
        # 标准COLMAP格式：前8字节是两个uint32（宽高各4字节）
        width = struct.unpack('<I', f.read(4))[0]  # 正确读取宽度
        height = struct.unpack('<I', f.read(4))[0] # 正确读取高度
        
        # 计算总像素数（添加安全校验）
        if width > 10000 or height > 10000:  # 防止异常值
            raise ValueError(f"Unreasonable dimensions: {width}x{height}")
            
        # 读取剩余数据（每个像素包含depth+confidence两个float32）
        data = np.frombuffer(f.read(), dtype=np.float32)
        
        # 验证数据长度（必须等于2*width*height）
        expected_length = 2 * width * height
        if len(data) != expected_length:
            raise ValueError(
                f"Data mismatch: Got {len(data)} floats, "
                f"expected {expected_length} (for {width}x{height})"
            )
        
        # 重组为(height, width, 2)的三维数组
        depth_map = data.reshape((height, width, 2))
        return depth_map[:, :, 0], depth_map[:, :, 1]  # 深度图, 置信度图

# 使用示例
depth, confidence = read_geometric_bin("./CC/stereo/depth_maps/frame0009.png.geometric.bin")
print(f"Depth map shape: {depth.shape}")
print(f"Confidence map shape: {confidence.shape}")