"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d
from typing import List, Tuple


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        for disparity in disparity_values:
            # right image is shifted 'opposite' the sign of the disparity in compliance with np.roll direction
            shifted_right_image = np.roll(right_image, -disparity, axis=1)
            if disparity < 0:  # right image is shifted to the right, left columns are zero-padded
                shifted_right_image[:, :-disparity, :] = 0
            else:  # right image is shifted to the left, right columns are zero-padded
                shifted_right_image[:, num_of_cols - disparity:, :] = 0

            ssdd_matrix = left_image - shifted_right_image
            ssdd_matrix = np.sum(ssdd_matrix ** 2, axis=2)
            ssdd_tensor[:, :, disparity - np.min(disparity_values)] = \
                convolve2d(ssdd_matrix, np.ones((win_size, win_size)), mode='same')

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        if num_of_cols == 1:
            return c_slice

        for column in range(num_of_cols):
            prev_column = l_slice[:, column - 1]
            optimal_route_cost = Solution.compute_optimal_route_cost(num_labels, num_of_cols, p1, p2, prev_column)
            l_slice[:, column] = c_slice[:, column] + optimal_route_cost - np.min(prev_column)

        return l_slice

    @staticmethod
    def compute_optimal_route_cost(num_labels: int,
                                   p1: float,
                                   p2: float,
                                   prev_column: np.ndarray) -> np.ndarray:
        disparity_labels = np.mgrid[range(num_labels), range(num_labels)]

        candidate_matrix = np.array([prev_column] * num_labels).reshape((num_labels, num_labels))
        candidate_matrix[abs(disparity_labels[0] - disparity_labels[1]) == 1] += p1
        candidate_matrix[abs(disparity_labels[0] - disparity_labels[1]) > 1] += p2

        return np.amin(candidate_matrix, axis=1)

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        for row_index in range(ssdd_tensor.shape[0]):
            l[row_index, :, :] = Solution.dp_grade_slice(ssdd_tensor[row_index, :, :].T, p1, p2).T

        return np.argmin(l, axis=2)

    @staticmethod
    def rotate_matrix_by_direction(matrix_to_rotate: np.ndarray,
                                   direction: int) -> np.ndarray:
        if direction % 2 == 1:  # row/column slices
            # if direction is 1/5- rows, else- columns and we transpose
            rotated_matrix = matrix_to_rotate if np.remainder(direction, 4) == 1 else matrix_to_rotate.transpose((1, 0, 2))
            # if direction is 5/7 the rows(/columns) are reversed, so flip left/right
            rotated_matrix = rotated_matrix if direction <= 4 else np.fliplr(rotated_matrix)
        else:  # diagonal slices
            # if direction is 6/8 the diagonals are bottom-row first, so flip up/down
            rotated_matrix = matrix_to_rotate if direction <= 4 else np.flipud(matrix_to_rotate)
            # if direction is 4/6 the diagonals are higher-column first, so flip left/right
            rotated_matrix = rotated_matrix if abs(direction - 5) > 2 else np.fliplr(rotated_matrix)

        return rotated_matrix

    def get_slices_by_direction(self,
                                tensor_to_slice: np.ndarray,
                                direction: int) -> List[np.ndarray]:
        rotated_tensor = Solution.rotate_matrix_by_direction(tensor_to_slice, direction)
        if direction % 2 == 1:  # row/column slices
            return [rotated_tensor[row_index, :, :].T for row_index in range(rotated_tensor.shape[0])]
        else:  # diagonal slices
            return [np.diagonal(rotated_tensor, offset=offset) for offset in
                    range(-rotated_tensor.shape[0] + 1, rotated_tensor.shape[1])]

    def create_label_matrix_from_slices(self,
                                        label_matrix_shape: Tuple[int],
                                        slice_values: List[np.ndarray],
                                        slice_index_list: List[np.ndarray]) -> np.ndarray:
        l = np.zeros(label_matrix_shape)
        for slice_indices, slice_value in zip(slice_index_list, slice_values):
            l[slice_indices[0], slice_indices[1]] = slice_value.T
        return l


    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        meshgrid = np.mgrid[range(l.shape[0]), range(l.shape[1])]
        meshgrid = np.moveaxis(meshgrid, 0, -1)
        for direction in range(1, num_of_directions + 1):
            slices_by_direction = self.get_slices_by_direction(ssdd_tensor, direction)
            slice_indices_by_direction = self.get_slices_by_direction(meshgrid, direction)
            label_slices = [Solution.dp_grade_slice(ssdd_slice, p1, p2) for ssdd_slice in slices_by_direction]
            l = self.create_label_matrix_from_slices(ssdd_tensor.shape, label_slices, slice_indices_by_direction,
                                                     direction, num_of_directions)

            direction_to_slice[direction] = np.argmin(l, 2)

        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        meshgrid = np.mgrid[range(l.shape[0]), range(l.shape[1])]
        meshgrid = np.moveaxis(meshgrid, 0, -1)
        for direction in range(1, num_of_directions + 1):
            slices_by_direction = self.get_slices_by_direction(ssdd_tensor, direction)
            slice_indices_by_direction = self.get_slices_by_direction(meshgrid, direction)
            label_slices = [Solution.dp_grade_slice(ssdd_slice, p1, p2) for ssdd_slice in slices_by_direction]
            l += self.create_label_matrix_from_slices(ssdd_tensor.shape, label_slices, slice_indices_by_direction,
                                                      direction, num_of_directions)
        l = l / num_of_directions
        return np.argmin(l, axis=2)
