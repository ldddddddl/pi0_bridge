import pyzed.sl as sl
import cv2
import numpy as np


class Camera:
    def __init__(self):
        self.zed = sl.Camera()
        self.dev = self.zed.get_device_list()[0]
        self.SN = self.dev.serial_number
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER

        self.opened = False
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        intrinsic_file_path = "/usr/local/zed/settings/SN{}.conf".format(self.SN)
        self.color_intrinsics_mat, self.distCoeffs = self.get_left_fhd_intrinsics(intrinsic_file_path)

        print("Camera intrinsics matrix:")
        print(self.color_intrinsics_mat)
        print("Camera distortion coefficients : ", self.distCoeffs)
        self.start()

    def get_left_fhd_intrinsics(self, file_path):
        """
        输入：文件路径
        输出：[LEFT_CAM_FHD]的内参矩阵（3x3）和畸变参数列表[k1, k2, p1, p2, k3]
        """
        params = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()

        in_section = False
        for line in lines:
            line = line.strip()
            if line == '[LEFT_CAM_FHD]':
                in_section = True
                continue
            if in_section:
                if line.startswith('[') and line.endswith(']'):  # 新段落开始
                    break
                if '=' in line:
                    k, v = line.split('=')
                    params[k.strip()] = float(v.strip())

        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']
        k1 = params['k1']
        k2 = params['k2']
        p1 = params['p1']
        p2 = params['p2']
        k3 = params['k3']

        K = [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
        dist = [k1, k2, p1, p2, k3]
        return np.asarray(K), np.asarray(dist)

    def start(self):
        try:
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ValueError(f"ZED Camera open failed: {err}")
            self.opened = True
            print("Camera started successfully.")

        except Exception as e:
            print(f"Failed to start camera: {str(e)}")
            raise

    def get_image(self):
        if not self.opened:
            return None, None

        try:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                image_np = self.image.get_data()[..., :3]

                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                depth_np = self.depth.get_data()

                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
                pcd_np = self.point_cloud.get_data()
                pcd_np[np.isnan(pcd_np)] = 0
                pcd_np = pcd_np[..., :3]

                return image_np, depth_np, pcd_np
            else:
                print("Failed to grab frame from camera.")
                return None, None
        except Exception as e:
            print(f"Error getting camera image: {str(e)}")
            return None, None

    def stop(self):
        try:
            if self.opened:
                self.zed.close()
                cv2.destroyAllWindows()
                self.opened = False
                print("Camera stopped successfully.")
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")

    def display_images(self, vis_type=""):
        if vis_type == "pointcloud":
            import open3d as o3d
        while self.opened:
            if vis_type == "pointcloud":
                color_image, depth_image, pcd_np = self.get_image()
                color_image = color_image[:,:,::-1]     # 调整通道为rgb顺序
                # 转换点云和颜色形状
                pcd_flat = pcd_np.reshape(-1, 3)
                rgb_flat = color_image.reshape(-1, 3) / 255.0

                # 创建点云对象
                pointcloud = o3d.geometry.PointCloud()
                pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
                pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

                zmin = 0.2
                zmax = 2.0

                points = np.asarray(pointcloud.points)
                mask = (points[:, 2] > zmin) & (points[:, 2] < zmax)
                cropped_points = points[mask]
                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
                # 如果原本有颜色，也要mask一下颜色
                if pointcloud.has_colors():
                    colors = np.asarray(pointcloud.colors)
                    cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

                o3d.visualization.draw_geometries([cropped_pcd])
                o3d.io.write_point_cloud("zed_pcd.ply", cropped_pcd)
            else:
                color_image, depth_image, pcd_np = self.get_image()
                if color_image is None or color_image is None:
                    continue

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=255 / 5.0),
                    cv2.COLORMAP_JET
                )

                # Create a side-by-side display
                images = np.hstack((color_image[:,:,:3], depth_colormap))

                # Show the images
                cv2.namedWindow('RealSense Camera Demo', cv2.WINDOW_NORMAL)
                cv2.imshow('RealSense Camera Demo', images)

                # Print the camera intrinsics
                if cv2.waitKey(1) == ord('i'):
                    intrinsics = camera.get_camera_intrinsics()
                    print("Camera Intrinsics:")
                    print(f"fx: {intrinsics['fx']}, fy: {intrinsics['fy']}")
                    print(f"px: {intrinsics['px']}, py: {intrinsics['py']}")

                    # Break the loop with 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break


if __name__ == "__main__":
    camera = Camera()
    vis_type = "pointcloud"
    try:
        camera.display_images(vis_type=vis_type)
    finally:
        camera.stop()