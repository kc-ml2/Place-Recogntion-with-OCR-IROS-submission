import argparse
import os

import cv2
from sklearn.preprocessing import minmax_scale
import numpy as np
from paddleocr import PaddleOCR
import textdistance as td

from environment.depth_sensor import DepthSensor
from localization.ocr_gaussian_map import OCRProbabilityMap
from localization.particle import Particle
from utils.config_import import load_config_module
from utils.map_utils import open_store_info, sparse_to_dense_grid
from utils.visualization import init_map_display, display_weighted_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fourview_realworld_similarity_adding.py")
    # parser.add_argument("--config", default="config/fourview_realworld_similarity.py")
    parser.add_argument("--visualize", action="store_true")
    args, _ = parser.parse_known_args()
    module_name = args.config
    is_visualize = args.visualize

    # Open files
    config = load_config_module(module_name)
    np.random.seed(config.DataConfig.NUMPY_SEED)

    # Load all navigable locations
    free_area = np.genfromtxt("./data/free_area_dense.csv", delimiter=",")

    # Open store location DB
    store_list, store_points, num_stores = open_store_info()

    # Get list of observation images
    # query_dir = "./data/query_imgs/"
    query_dir = "./data/query_imgs_copy/"
    query_list = sorted(os.listdir(query_dir))
    num_queries = len(query_list)
    query_idx = np.arange(start=0, stop=num_queries, step=4)

    # Initialize OCR, Depth estimation, Object probability map and Particles for Monte Carlo localzation
    ocr_en = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    depth_sensor = DepthSensor(config)
    particles = Particle(config, manual_pos_arr=free_area)
    probability_map = OCRProbabilityMap(config, particles, store_points)

    total_distance_list = []

    if is_visualize:
        img_path = "./data/picmap(HD_scale).png"
        map_img = cv2.imread(img_path)
        init_map_display()

    for id in query_idx:
        query_pos_str = query_list[id][:6]
        print("Image: ", query_pos_str)

        img_list = []
        box_result = [[] for _ in range(4)]
        word_result = [[] for _ in range(4)]

        # Iterate four images: left, front, right, back
        for j in range(4):
            img_path = query_dir + query_list[id + j]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)

            # Extract OCR data
            ocr_data = ocr_en.ocr(img, cls=True)

            if ocr_data[0] is not None:
                for item in ocr_data[0]:
                    box = np.array(item[0], dtype=np.int64)
                    box = np.transpose([box[:, 1], box[:, 0]])  # y,x
                    box_mid_point = np.int64(np.average(box, axis=0))
                    word = item[1][0].replace(" ", "").lower()

                    box_result[j].append(box_mid_point)
                    word_result[j].append(word)

        # Make observation dictionary to get the depth images
        observations = {}
        observations["left_view"] = img_list[0]
        observations["front_view"] = img_list[1]
        observations["right_view"] = img_list[2]
        observations["back_view"] = img_list[3]

        depth_img_batch = depth_sensor.get_depth_batch(observations)

        all_words = sum(word_result, [])
        similarity_mat = np.zeros([len(all_words), num_stores])

        # Pre-calculate the similarity measure between 'detected words' and 'store names'
        for i, single_word in enumerate(all_words):
            for j, store in enumerate(store_list):
                store_lowered = store.replace(" ", "").lower()
                similarity_mat[i][j] = td.levenshtein.normalized_similarity(store_lowered, single_word)

        # Run localization algorithm with suggested method
        if config.DataConfig.USE_ADDING_PROB:
            result_particle_id, probability_by_particle = probability_map.localize_with_ocr_adding(
            # result_particle_id, probability_by_particle = probability_map.localize_with_ocr_baseline(
            # result_particle_id, probability_by_particle = probability_map.localize_with_ocr_baseline2(
                particles, depth_sensor, box_result, depth_img_batch, similarity_mat
            )
        else:
            result_particle_id, probability_by_particle = probability_map.localize_with_ocr(
                particles, depth_sensor, box_result, depth_img_batch, similarity_mat
            )

        # if is_visualize:
        #     positions = np.delete(particles.translations, 1, axis=1)
        #     # probability_by_particle = np.log(probability_by_particle)
        #     normalized_list = minmax_scale(probability_by_particle)
        #     normalized_array = np.expand_dims(np.array(normalized_list), axis=1)
        #     weighted_particles = np.concatenate([positions, normalized_array], axis=1)

        #     display_weighted_map(map_img, weighted_particles)
        
        if is_visualize:
            positions = np.delete(particles.translations, 1, axis=1)
            # probability_by_particle = np.log(probability_by_particle)
            probability_by_particle = np.expand_dims(np.array(probability_by_particle), axis=1)
            weighted_particles = np.concatenate([positions, probability_by_particle], axis=1)
            sorted_weighted_particles = weighted_particles[weighted_particles[:, 2].argsort()[::-1]]
            
            id = 0
            for i, value in enumerate(sorted_weighted_particles):
                if value[2] != 1.0:
                    id = i
                    break
            sorted_weighted_particles = sorted_weighted_particles[id:]

            weight_list = sorted_weighted_particles[:, 2]
            normalized_list = minmax_scale(weight_list)
            normalized_array = np.expand_dims(np.array(normalized_list), axis=1)

            weighted_particles = np.concatenate([sorted_weighted_particles[:, :2], normalized_array], axis=1)

            display_weighted_map(map_img, weighted_particles)

        # Get estimated position. Project to 2D floor plan, and convert coordinate
        result_position = particles.translations[result_particle_id]  # x,y,z
        projected_result_pos = np.delete(result_position, 1)  # x,z
        projected_result_pos = np.roll(projected_result_pos, 1)  # z,x
        sparse_query_pos = np.array([int(query_pos_str[0:3]), int(query_pos_str[3:6])])  # z,x
        sparse_query_pos = np.expand_dims(sparse_query_pos, axis=0)
        query_pos = sparse_to_dense_grid(sparse_query_pos)

        distance = np.linalg.norm(projected_result_pos - query_pos) * depth_sensor.meters_per_pixel
        total_distance_list.append(distance)

        print("Distance from GT and Estimated pos: ", distance)

    print("Final result: ", sum(total_distance_list) / len(total_distance_list))

    total_distance_arr = np.array(total_distance_list)
    print("Number of samples: ", len(total_distance_arr))
    print("Final accuracy 20.0: ", len(total_distance_arr[total_distance_arr <= 20.0]) / len(total_distance_arr))
    print("Final accuracy 10.0: ", len(total_distance_arr[total_distance_arr <= 10.0]) / len(total_distance_arr))
