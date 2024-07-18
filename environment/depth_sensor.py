import math

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class DepthSensor:
    def __init__(self, config):
        self.num_camera = config.CamConfig.NUM_CAMERA
        self.sensor_height = config.CamConfig.SENSOR_HEIGHT
        self.meters_per_pixel = config.DataConfig.METERS_PER_PIXEL

        # Parameter for depth image
        self.img_h = config.CamConfig.HEIGHT
        self.img_w = config.CamConfig.WIDTH
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0
        self.cxy = np.array([self.cx, self.cy])
        self.fx = self.cx / np.tan(math.radians(90) / 2.0)
        self.fy = self.cy / np.tan(math.radians(90) / 2.0)
        self.fxy = np.array([self.fx, self.fy])

        # Image frame for depth image to point cloud conversion
        x_axis = np.arange(0, self.img_w)
        x_axis = np.flip(x_axis)
        y_axis = np.arange(0, self.img_h)
        y_axis = np.flip(y_axis)
        y_axis = y_axis[:, np.newaxis]  # 1-D array tranpose

        self.x_frame = np.broadcast_to(x_axis, (self.img_h, self.img_w))
        self.y_frame = np.broadcast_to(y_axis, (self.img_h, self.img_w))
        self.xy_frame = np.stack([self.x_frame, self.y_frame], axis=2)

        depth_model_type = "DPT_BEiT_L_512"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.depth_scale = 5000.0
        self.midas = torch.hub.load("intel-isl/MiDaS", depth_model_type)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def get_depth_batch(self, observations):
        img_vector = np.array([])
        obs = observations

        if self.num_camera == 4:
            img_vector = np.array([obs["left_view"], obs["front_view"], obs["right_view"], obs["back_view"]])
        if self.num_camera == 3:
            img_vector = np.array([obs["left_view"], obs["front_view"], obs["right_view"]])
        if self.num_camera == 1:
            img_vector = np.array([obs["all_view"]])

        # Get estimated inverse relative depth & convert it to absolute depth
        depth_img_batch = self.estimate_depth(img_vector)
        depth_img_batch = 1.0 / depth_img_batch
        depth_img_batch = depth_img_batch * self.depth_scale

        return depth_img_batch

    def estimate_depth(self, img_batch):
        img_shape = img_batch[0].shape[:2]
        input_batch = torch.cat([self.transform(img).to(self.device) for img in img_batch])

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        return output

    def depth_point2pcd(self, pixel_points: np.ndarray, depth_values: np.ndarray):
        """Calculate positions of point cloud points from depth image."""
        # Get flipped camera sensor coordinates
        rows = pixel_points[:, 0]
        cols = pixel_points[:, 1]

        frame_points = self.xy_frame[rows, cols, :]

        xy = (frame_points - self.cxy) / self.fxy * depth_values[:, np.newaxis]
        xyz = np.concatenate([xy, depth_values[:, np.newaxis]], axis=1)

        return xyz

    @staticmethod
    def get_depth_by_pixel(depth_img, pixel_points: np.ndarray, offset=0.5):
        rows = pixel_points[:, 0]
        cols = pixel_points[:, 1]

        depth_values = depth_img[rows, cols] + offset

        return depth_values

    def multiview_box_centers_to_firstview_points(self, box_center_batch, depth_img_batch):
        first_view_object_points = np.array([], dtype=np.float64).reshape(0, 3)

        for frame_id, box_centers in enumerate(box_center_batch):
            if len(box_centers) == 0:
                continue

            box_arr = np.stack(box_centers, axis=0)
            depth_values = self.get_depth_by_pixel(depth_img_batch[frame_id], box_arr, offset=0.0)
            pcd_points = self.depth_point2pcd(box_arr, depth_values)

            frame_rotation = (1 - frame_id) * 90
            r = R.from_euler("y", frame_rotation, degrees=True)
            pcd_points = r.apply(pcd_points)

            first_view_object_points = np.concatenate([first_view_object_points, pcd_points])

        first_view_object_points[:, 1] = first_view_object_points[:, 1] + self.sensor_height
        first_view_object_points = first_view_object_points / self.meters_per_pixel

        return first_view_object_points
