import csv

import cv2
import numpy as np


def sparse_to_dense_grid(grid_arr, unit_pixel=92.75, z_offset=191, x_offset=273):
    """Convert from real value (sparse) based position to map image (dense) based position."""
    grid_dense = grid_arr * unit_pixel + unit_pixel / 2
    grid_dense[:, 0] = np.round(grid_dense[:, 0] + z_offset)
    grid_dense[:, 1] = np.round(grid_dense[:, 1] + x_offset)

    return grid_dense


def open_store_info(csv_path="./data/store_info_eng.csv"):
    """Open store name location database."""
    with open(csv_path, newline="") as csvfile:  # pylint: disable=unspecified-encoding
        len_store = len(csvfile.readlines())

    with open(csv_path, newline="") as csvfile:  # pylint: disable=unspecified-encoding
        reader = csv.DictReader(csvfile)
        db_store_list = []
        db_store_points = np.zeros([len_store, 2], dtype=np.int64)

        for i, row in enumerate(reader):
            db_store_list.append(row["name"])
            db_store_points[i] = [row["z"], row["x"]]

    return db_store_list, db_store_points, len_store


def remove_isolated_area(topdown_map, removal_threshold=1000):
    """Remove isolated small area to avoid unnecessary graph node."""
    contours, _ = cv2.findContours(topdown_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < removal_threshold:
            cv2.fillPoly(topdown_map, [contour], 0)

    return topdown_map
