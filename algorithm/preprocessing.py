import numpy as np
from typing import Tuple

import cv2
from sklearn.decomposition import PCA

from algorithm.basic import sd_filter, fill_hole, fg_pts, get_angle, \
    saturation_rectified_intensity, rescale_foreground


def extract(image: np.ndarray, kernel: np.ndarray, thr: float) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    :param image: 3D numpy array
    :param kernel: 2-element tuple
    :param thr: minimum sd
    :return: (3D numpy array, 2D numpy array)
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_sd = sd_filter(image_gray, kernel)
    image_thr = cv2.threshold(image_sd, thr, 255,
                              cv2.THRESH_BINARY)[1].astype(np.uint8)
    mask_out = fill_hole(image_thr)
    mask_color = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    image_out = cv2.bitwise_or(image, cv2.bitwise_not(mask_color))
    return image_out, mask_out


def affine_correct(image: np.ndarray, mask: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    :param image: grayscale image, 2D numpy array
    :param mask: 2D binary numpy array
    :return: (3D numpy array, 2D numpy array)
    """
    # Make x, y axis correspond to the width, height direction.
    pts = np.flip(fg_pts(mask))
    # PCA to determine the main direction and construct the transform matrix.
    pca = PCA(2)
    pca.fit(pts)
    angle = get_angle(pca.components_[0, :], [1, 0])
    affine_mat = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    # Get transformed foreground points to find their x, y limit for translating.
    new_pts = np.dot(affine_mat[:, :2], pts.transpose())
    x_min, x_max = new_pts[0, :].min(), new_pts[0, :].max()
    y_min, y_max = new_pts[1, :].min(), new_pts[1, :].max()
    width, height = int(x_max - x_min), int(y_max - y_min)
    affine_mat[:, 2] = [-x_min, -y_min]
    # Apply affine transformation.
    image_out = cv2.warpAffine(image, affine_mat,
                               (width, height), borderValue=255)
    mask_out = cv2.warpAffine(mask, affine_mat,
                              (width, height), borderValue=0)
    return image_out, mask_out


def register(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    :param image: 3D numpy array
    :param mask: 2D numpy array
    :param size: (width, height)
    :return: (3D numpy array, 2D numpy array)
    """
    # Convert to grayscale, extract staining pixels, and rescale intensity.
    image_stain = saturation_rectified_intensity(image)
    image_rescale = rescale_foreground(image_stain, mask)
    # Rotation and resize.
    image_rot, mask_rot = affine_correct(image_rescale, mask)
    image_out = cv2.resize(image_rot, size)
    mask_out = cv2.resize(mask_rot, size)
    return image_out, mask_out
