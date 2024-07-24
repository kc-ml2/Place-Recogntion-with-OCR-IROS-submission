import numpy as np


class OCRProbabilityMap:
    """Class for semantic localization with Object probability map + similarity measure + OCR."""

    def __init__(self, config, particle, store_points, sigma=np.array([[20.0, 0.0], [0.0, 20.0]])):
        self.config = config

        store_points = np.roll(store_points, 1, axis=1)
        self.total_object_list = np.insert(store_points, 1, 0, axis=1)
        self.sigma_det = np.linalg.det(sigma)
        self.sigma_inv = np.linalg.inv(sigma)

        self.mu_particle_matrix_on_firstview = self.mu_particle_matrix_map_obj_on_firstview(particle)

    def gaussian_mixture_kernel(self, target_mu: np.ndarray, anchor_mu_batch: np.ndarray):
        """Return the multiple Gaussian kernel results on a single position array."""
        if len(target_mu.shape) == 1:  # if there is only one data
            if target_mu.shape[0] == 3:  # if data is 3D point
                target_mu = np.delete(target_mu, 1)  # it takes only 2D point data
        else:
            raise ValueError("Input for target_mu must be a single vector.")

        N = np.sqrt((2 * np.pi) ** 2 * self.sigma_det)

        num_particle = anchor_mu_batch.shape[0]
        num_anchor_object = anchor_mu_batch.shape[1]
        target_batch = np.full([num_particle, num_anchor_object, 2], target_mu)
        x_minus_mu = target_batch - anchor_mu_batch

        # Calculates (x-mu)T.sigma-1.(x-mu)
        fac = np.einsum("...i,...i->...", x_minus_mu.dot(self.sigma_inv), x_minus_mu)
        results = np.exp(-1 * fac / 2) / N

        return results

    def mu_particle_matrix_map_obj_on_firstview(self, particle_instance, visible_distance=300.0):
        """Return transformed positions of total object on the current level.
        Transform: map coordinate -> first view coordinate."""
        total_mu_array = self.total_object_list
        total_projected_transformed_mu = np.zeros([particle_instance.num_particles, 0, 2], dtype=np.float64)

        for object_mu in total_mu_array:
            object_mu = np.concatenate([object_mu, [1.0]])
            mu_on_particle = np.matmul(particle_instance.inverse_transform_mat, object_mu)[:, :3]

            projected_mu_on_particle = np.delete(mu_on_particle, [1], axis=1)
            projected_mu_on_particle = np.expand_dims(projected_mu_on_particle, axis=1)

            total_projected_transformed_mu = np.concatenate(
                [total_projected_transformed_mu, projected_mu_on_particle],
                axis=1,
            )

        # Convert distant (invisiable) object values to np.nan
        x_points = total_projected_transformed_mu[:, :, 0]
        z_points = total_projected_transformed_mu[:, :, 1]
        invisible_idx = np.where(np.sqrt(x_points**2 + z_points**2) > visible_distance)
        total_projected_transformed_mu[invisible_idx] = np.nan

        firstview_result_by_object = total_projected_transformed_mu

        return firstview_result_by_object

    def localize_with_ocr(self, particle_instance, depth_sensor, box_center_batch, depth_img_batch, similarity_mat):
        """Conduct localization based on detected objects.
        Compare positions of each detected objects with positions of objects in map."""
        # Generate detected object list to iterate
        # Merge detected words of four view(left, front, right, back) into one array
        current_mu_array = depth_sensor.multiview_box_centers_to_firstview_points(box_center_batch, depth_img_batch)

        mat_for_product = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)

        for i, single_mu in enumerate(current_mu_array):
            # Check if there are similary words on map
            similar_word_idx = np.where(similarity_mat[i] >= self.config.DataConfig.SIMILARITY_THRESHOLD)[0]

            # Calculate probabilities of current word location with all OPMs of similar words
            mat_for_similar_group = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)
            for word_idx in similar_word_idx:
                single_word_mat = self.mu_particle_matrix_on_firstview[:, word_idx, :]
                single_word_mat = np.expand_dims(single_word_mat, axis=1)

                if np.shape(single_word_mat)[1] != 0:
                    gaussian_result = self.gaussian_mixture_kernel(single_mu, single_word_mat)
                    gaussian_result = gaussian_result * similarity_mat[i][word_idx]

                    mat_for_similar_group = np.concatenate([mat_for_similar_group, gaussian_result], axis=1)

            # Get the maximum probability
            if np.shape(mat_for_similar_group)[1] != 0:
                max_value = np.nanmax(mat_for_similar_group, axis=1)
                max_value = np.expand_dims(max_value, axis=1)
                mat_for_product = np.concatenate([mat_for_product, max_value], axis=1)

        mat_for_product = np.ma.array(mat_for_product, mask=np.isnan(mat_for_product))
        num_considered_objects = np.shape(mat_for_product)[1]

        # If there is no detected word, pick a random position
        if num_considered_objects == 0:
            result_particle_id = np.random.randint(0, particle_instance.num_particles)
            probability_by_particle = np.ones([particle_instance.num_particles])
        # Product probabilities of each words
        else:
            num_mask = mat_for_product.mask.sum(axis=1)
            num_nonnan = np.full(particle_instance.num_particles, num_considered_objects) - num_mask

            temp_numerator_array = np.full(particle_instance.num_particles, 1, dtype=np.float64)
            nth_root = np.divide(
                temp_numerator_array,
                num_nonnan,
                out=np.full(np.shape(temp_numerator_array), 1, dtype=np.float64),
                where=num_nonnan != 0,
            )

            probability_by_particle = np.exp(np.sum(np.log(mat_for_product), axis=1))
            probability_by_particle = np.power(probability_by_particle, nth_root)
            result_particle_id = np.argmax(probability_by_particle)

        return result_particle_id, probability_by_particle

    def localize_with_ocr_baseline(self, particle_instance, depth_sensor, box_center_batch, depth_img_batch, similarity_mat):
        """Conduct localization based on detected objects.
        Compare positions of each detected objects with positions of objects in map."""
        # Generate detected object list to iterate
        # Merge detected words of four view(left, front, right, back) into one array
        current_mu_array = depth_sensor.multiview_box_centers_to_firstview_points(box_center_batch, depth_img_batch)

        mat_for_product = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)

        for i, single_mu in enumerate(current_mu_array):
            # Check if there are similary words on map
            similar_word_idx = np.where(similarity_mat[i] >= self.config.DataConfig.SIMILARITY_THRESHOLD)[0]

            # Calculate probabilities of current word location with all OPMs of similar words
            mat_for_similar_group = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)
            for word_idx in similar_word_idx:
                single_word_mat = self.mu_particle_matrix_on_firstview[:, word_idx, :]
                single_word_mat = np.expand_dims(single_word_mat, axis=1)

                if np.shape(single_word_mat)[1] != 0:
                    gaussian_result = self.gaussian_mixture_kernel(single_mu, single_word_mat)
                    gaussian_result = gaussian_result * similarity_mat[i][word_idx]

                    mat_for_similar_group = np.concatenate([mat_for_similar_group, gaussian_result], axis=1)

            # Get the sum of probabilities from similar group
            if np.shape(mat_for_similar_group)[1] != 0:
                probability_of_group = np.sum(mat_for_similar_group, axis=1)
                probability_of_group = np.expand_dims(probability_of_group, axis=1)
                mat_for_product = np.concatenate([mat_for_product, probability_of_group], axis=1)

        mat_for_product = np.ma.array(mat_for_product, mask=np.isnan(mat_for_product))
        num_considered_objects = np.shape(mat_for_product)[1]

        # If there is no detected word, pick a random position
        if num_considered_objects == 0:
            result_particle_id = np.random.randint(0, particle_instance.num_particles)
            probability_by_particle = np.ones([particle_instance.num_particles])
        # Product probabilities of each words
        else:
            num_mask = mat_for_product.mask.sum(axis=1)
            num_nonnan = np.full(particle_instance.num_particles, num_considered_objects) - num_mask

            temp_numerator_array = np.full(particle_instance.num_particles, 1, dtype=np.float64)
            nth_root = np.divide(
                temp_numerator_array,
                num_nonnan,
                out=np.full(np.shape(temp_numerator_array), 1, dtype=np.float64),
                where=num_nonnan != 0,
            )

            probability_by_particle = np.exp(np.sum(np.log(mat_for_product), axis=1))
            probability_by_particle = np.power(probability_by_particle, nth_root)
            result_particle_id = np.argmax(probability_by_particle)

        return result_particle_id, probability_by_particle

    def localize_with_ocr_adding(
        self, particle_instance, depth_sensor, box_center_batch, depth_img_batch, similarity_mat
    ):
        """Conduct localization based on detected objects.
        Compare positions of each detected objects with positions of objects in map."""
        # Generate detected object list to iterate
        # Merge detected words of four view(left, front, right, back) into one array
        current_mu_array = depth_sensor.multiview_box_centers_to_firstview_points(box_center_batch, depth_img_batch)

        mat_for_add = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)

        for i, single_mu in enumerate(current_mu_array):
            # Check if there are similary words on map
            similar_word_idx = np.where(similarity_mat[i] >= self.config.DataConfig.SIMILARITY_THRESHOLD)[0]

            # Calculate probabilities of current word location with all OPMs of similar words
            mat_for_similar_group = np.zeros([particle_instance.num_particles, 0], dtype=np.float64)
            for word_idx in similar_word_idx:
                single_word_mat = self.mu_particle_matrix_on_firstview[:, word_idx, :]
                single_word_mat = np.expand_dims(single_word_mat, axis=1)

                if np.shape(single_word_mat)[1] != 0:
                    gaussian_result = self.gaussian_mixture_kernel(single_mu, single_word_mat)
                    gaussian_result = gaussian_result * similarity_mat[i][word_idx]

                    mat_for_similar_group = np.concatenate([mat_for_similar_group, gaussian_result], axis=1)

            # Get the maximum probability
            if np.shape(mat_for_similar_group)[1] != 0:
                max_value = np.nanmax(mat_for_similar_group, axis=1)
                max_value = np.expand_dims(max_value, axis=1)
                mat_for_add = np.concatenate([mat_for_add, max_value], axis=1)

        mat_for_add = np.ma.array(mat_for_add, mask=np.isnan(mat_for_add))
        num_considered_objects = np.shape(mat_for_add)[1]

        # If there is no detected word, pick a random position
        if num_considered_objects == 0:
            result_particle_id = np.random.randint(0, particle_instance.num_particles)
            probability_by_particle = np.ones([particle_instance.num_particles])
        # Product probabilities of each words
        else:
            probability_by_particle = np.sum(mat_for_add, axis=1)
            result_particle_id = np.argmax(probability_by_particle)

        return result_particle_id, probability_by_particle
