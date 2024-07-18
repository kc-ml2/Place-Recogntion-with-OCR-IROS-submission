import cv2
import numpy as np


def init_opencv_cam(x_size=1152, y_size=864):
    """Create image window for observation."""
    cv2.namedWindow("observation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("observation", x_size, y_size)
    cv2.moveWindow("observation", 1352, 20)


def display_opencv_cam(rgb_obs) -> int:
    """Draw nodes and edges into map image."""
    cv2.imshow("observation", rgb_obs)
    key = cv2.waitKey()

    return key


def init_map_display(window_name="map", x_size=1152, y_size=864):
    """Create image window for map."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, x_size, y_size)
    cv2.moveWindow(window_name, 10, -20)


def display_map(topdown_map, window_name="map", key_points=None, wait_for_key=False):
    """Display a topdown map with OpenCV."""

    if key_points is not None:
        for pnt in key_points:
            cv2.drawMarker(
                img=topdown_map,
                position=(int(pnt[1]), int(pnt[0])),
                color=(255, 0, 0),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=1,
            )

    if key_points is not None and len(key_points) >= 2:
        for i in range(len(key_points) - 1):
            cv2.line(
                img=topdown_map,
                pt1=(int(key_points[i][1]), int(key_points[i][0])),
                pt2=(int(key_points[i + 1][1]), int(key_points[i + 1][0])),
                color=(0, 255, 0),
                thickness=1,
            )

    cv2.imshow(window_name, topdown_map)
    if wait_for_key:
        cv2.waitKey()


def draw_point_from_grid_pos(map_img, grid_pos, color, size=1, thickness=-1):
    """Draw highlighted point(circle) on map with pixer position."""
    dimension_of_data = len(np.shape(grid_pos))
    img_h, img_w, _ = np.shape(map_img)

    if dimension_of_data == 1:
        cv2.circle(
            img=map_img,
            center=(int(grid_pos[1]), int(grid_pos[0])),
            radius=size,
            color=color,
            thickness=thickness,
        )

    elif dimension_of_data == 2:
        grid_pos = grid_pos.astype(np.int64)
        grid_z = np.expand_dims(grid_pos[:, 0], axis=1)
        grid_z[grid_z >= img_h] = img_h - 1
        grid_z[grid_z < 0] = 0
        grid_x = np.expand_dims(grid_pos[:, 1], axis=1)
        grid_x[grid_x >= img_w] = img_w - 1
        grid_x[grid_x < 0] = 0

        grid_pos = np.concatenate([grid_z, grid_x], axis=1)
        map_img[grid_pos[:, 0], grid_pos[:, 1]] = color

    else:
        print("Dimension of grid position data is not suited for drawing.")
