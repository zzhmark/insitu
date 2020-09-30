import numpy as np
from typing import List

import cv2


def fgPts(mask: np.ndarray):
    """
    :param mask: binary image, 2D numpy array
    :return: 2 * n numpy array
    Retrieves coordinates of points whose intensity is more than 0.
    """
    height, width = mask.shape
    pts_list = [[i, j]
                for i in range(height)
                for j in range(width)
                if mask[i, j] > 0]
    return np.array(pts_list)


def angle_of_vectors(v1: List[int], v2: List[int]):
    """
    :param v1: 2D vector
    :param v2: 2D vector
    :return: the angle of v1 and v2 in degrees
    """
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot / norm))


def std_dev_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    :param image: grayscale image, 2D numpy array
    :param kernel: 2-element tuple
    :return: 2D numpy float array
    Calculate standard deviation on an array, based on a specific kernel.
    """
    a = image.astype(np.float)
    return cv2.sqrt(cv2.blur(a ** 2, kernel) - cv2.blur(a, kernel) ** 2)


def fillHole(mask: np.ndarray) -> np.ndarray:
    """
    :param mask: grayscale image, 2D numpy array
    :return: binary image, 2D numpy array
    Find contours in an array image, fill the out most one.
    """
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(mask, contour, -1, 255, cv2.FILLED)


def sigmoid(x: np.ndarray, cutoff: float, gain: float) -> np.ndarray:
    """
    :param x: the array of input
    :param cutoff: the midpoint of threshold
    :param gain: the slope
    :return: 2D numpy array
    """
    return 1 / (1 + np.exp(gain * (cutoff - x)))


def adjust2gray(image: np.ndarray) -> np.ndarray:
    """
    :param image: bgr image, 3D numpy array
    :return: grayscale image, 2D numpy array
    Convert color image to grayscale, also apply sigmoid to weaken
    low saturation pixels
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    adjust = sigmoid(image_hsv[:, :, 1].astype(np.float), 20, 0.1)
    signal = (255 - image_hsv[:, :, 2])
    image_out = (255 - signal * adjust).astype(np.uint8)
    return image_out
