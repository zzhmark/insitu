import numpy as np
from typing import List, Any

import cv2
from skimage.exposure import rescale_intensity, adjust_sigmoid
from skimage.util import invert, img_as_float, img_as_ubyte


def fg_pts(mask: np.ndarray):
    """
    :param mask: binary image, 2D numpy array
    :return: 2 * n numpy array
    Retrieves coordinates of points whose intensity is more than 0.
    """
    height, width = mask.shape
    pts = [[i, j]
           for i in range(height)
           for j in range(width)
           if mask[i, j] > 0]
    return np.asarray(pts)


def get_angle(v1: List[int], v2: List[int]):
    """
    :param v1: 2D vector
    :param v2: 2D vector
    :return: the angle of v1 and v2 in degree
    """
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot / norm))


def sd_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    :param image: grayscale image, 2D numpy array
    :param kernel: 2-element tuple
    :return: 2D numpy float array
    Calculate standard deviation on an array, based on a specific kernel.
    """
    a = image.astype(np.float)
    return cv2.sqrt(cv2.blur(a ** 2, kernel) - cv2.blur(a, kernel) ** 2)


def fill_hole(mask: np.ndarray) -> np.ndarray:
    """
    :param mask: grayscale image, 2D numpy array
    :return: binary image, 2D numpy array
    Find contours in an array image, fill the out most one.
    """
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(mask, contour, -1, 255, cv2.FILLED)


def saturation_rectified_intensity(image: np.ndarray) -> np.ndarray:
    """
    :param image: BGR image, 3D numpy array
    :return: grayscale image, 2D numpy array
    Convert color image to grayscale, also apply sigmoid to weaken
    low saturation pixels.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = img_as_float(image_hsv[:, :, 1])
    intensity = img_as_float(image_hsv[:, :, 2])
    adjust = adjust_sigmoid(saturation, 0.08, 25)
    signal = invert(intensity)
    image_out = invert(adjust * signal)
    return img_as_ubyte(image_out)


def rescale_foreground(image: Any, mask: np.ndarray) -> Any:
    """
    :param image: grayscale image, 2D numpy array
    :param mask: binary image, 2D numpy array
    :return: grayscale image, 2D numpy array
    Linearly transform a grayscale image so that its intensity spans [0, 255].
    """
    fg_intensity = image[mask > 0]
    fg_range = np.min(fg_intensity), np.max(fg_intensity)
    return rescale_intensity(image, fg_range)
