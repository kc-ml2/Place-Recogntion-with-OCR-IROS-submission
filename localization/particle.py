import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.map_utils import remove_isolated_area


class Particle:
    """Class for generating particles for Monte Carlo Localization."""

    def __init__(self, config, sim=None, level=None, manual_pos_arr=None):
        self.num_particles = config.DataConfig.NUM_PARTICLES

        if sim is None:
            grid_size = np.array([1, 1])
            corner = np.array([0, 0])
            height = 0

            all_explorable_grid_pos = manual_pos_arr

        else:
            if config.DataConfig.REMOVE_ISOLATED:
                binary_topdown_map = remove_isolated_area(sim.topdown_map_list[level])
            else:
                binary_topdown_map = sim.topdown_map_list[level]

            lower_bound, upper_bound = sim.pathfinder.get_bounds()
            grid_resolution = sim.topdown_map_list[level].shape[0:2]
            grid_size = np.array(
                [
                    abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
                    abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
                ]
            )
            corner = np.array([lower_bound[2], lower_bound[0]])
            height = sim.height_list[level]

            explorable_grid_pos_list = list(zip(*np.where(binary_topdown_map == 1)))
            all_explorable_grid_pos = np.array(explorable_grid_pos_list)

        rng = np.random.default_rng(seed=config.DataConfig.NUMPY_SEED)

        all_realworld_pos = all_explorable_grid_pos * grid_size + corner
        sampled_positions = rng.choice(all_realworld_pos, size=self.num_particles)

        sampled_rotations = rng.integers(0, 360, size=self.num_particles)
        sampled_rotations = np.expand_dims(sampled_rotations, axis=1)

        # Convert y-x to x-y
        sampled_positions = np.c_[
            sampled_positions[:, 1],
            np.full(self.num_particles, height),
            sampled_positions[:, 0],
        ]

        self.sampled_poses = np.concatenate([sampled_positions, sampled_rotations], axis=-1)
        self.translations = self.sampled_poses[:, :3]
        self.rotations = self.sampled_poses[:, 3]

        rotation_mat = R.from_euler("y", self.rotations, degrees=True).as_matrix()
        zeros_row = np.zeros([self.num_particles, 1, 3], dtype=np.float64)
        self.transform_mat = np.concatenate([rotation_mat, zeros_row], axis=1)
        position_column = np.full([self.num_particles, 4, 1], 1.0, dtype=np.float64)
        self.transform_mat = np.concatenate([self.transform_mat, position_column], axis=2)
        self.transform_mat[:, :3, 3] = self.translations

        self.inverse_transform_mat = self.transform_mat.copy()
        r_transpose = np.transpose(self.transform_mat[:, :3, :3], (0, 2, 1))
        self.inverse_transform_mat[:, :3, :3] = r_transpose
        inverse_dot_pos = np.einsum("...jk,...k", r_transpose, self.inverse_transform_mat[:, :3, 3])
        self.inverse_transform_mat[:, :3, 3] = inverse_dot_pos * -1.0
