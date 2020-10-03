import numpy as np

from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

from .basic import fg_pts


def blob_score(pts1: np.ndarray, pts2: np.ndarray,
               mean1: pd.Series, mean2: pd.Series):
    """
    :param pts1: blob1 points coordinate list
    :param pts2: blob2 points coordinate list
    :param mean1: the mean intensity of blob1
    :param mean2: the mean intensity of blob2
    :return: a score
    Calculate 2 blobs' similarity score, determined by
    the relative difference of their intensity and their
    relative overlap area.
    """
    grad_term = 1 - np.abs(mean1 - mean2) / 256
    set1, set2 = set(tuple(i) for i in pts1), set(tuple(i) for i in pts2)
    overlap_term = len(set1.intersection(set2)) / len(set2.union(set2))
    return grad_term * overlap_term


def local_score(label1: np.ndarray, label2: np.ndarray,
                means1: pd.Series, means2: pd.Series) -> float:
    """
    :param label1: labeled image, 2D numpy array
    :param label2: labeled image, 2D numpy array
    :param means1: means for points in different levels
    :param means2: means for points in different levels
    :return: the local score
    Calculate 2 images' similarity score, by working out similarity
    scores between any 2 blobs, and sum up the best matches.
    """
    n1, n2 = len(means1), len(means2)
    # List points for different levels.
    pts_list1 = [fg_pts(label1 == i + 1) for i in range(n1)]
    pts_list2 = [fg_pts(label2 == i + 1) for i in range(n2)]
    score_blobs = np.zeros((n1, n2))
    for i, pts1, mean1 in zip(range(n1), pts_list1, means1):
        for j, pts2, mean2 in zip(range(n2), pts_list2, means2):
            score_blobs[i, j] = blob_score(pts1, pts2, mean1, mean2)
    return np.max(score_blobs, axis=0).sum() + np.max(score_blobs, axis=1).sum()


def global_gmm_compare(mask1: np.ndarray, mask2: np.ndarray,
                       label1: np.ndarray, label2: np.ndarray):
    """
    :param mask1: the foreground region, 2D numpy array
    :param mask2: the foreground region, 2D numpy array
    :param label1: labeled image, 2D numpy array
    :param label2: labeled image, 2D numpy array
    :return: score
    Compare 2 masks in 4 orientations and return the best score.
    """
    # Four orientations for labels and masks.
    label1_flip_x = np.flip(label1, axis=0)
    label2_flip_y = np.flip(label2, axis=1)
    mask1_flip_x = np.flip(mask1, axis=0)
    mask2_flip_y = np.flip(mask2, axis=1)
    # Convert to 1D array.
    label1_1d = np.resize(label1, -1)
    label2_1d = np.resize(label2, -1)
    label1_flip_x_1d = np.resize(label1_flip_x, -1)
    label2_flip_y_1d = np.resize(label2_flip_y, -1)
    mask1_1d = np.resize(mask1, -1)
    mask2_1d = np.resize(mask2, -1)
    mask1_flip_x_1d = np.resize(mask1_flip_x, -1)
    mask2_flip_y_1d = np.resize(mask2_flip_y, -1)
    # Foregrounds.
    fg1 = (mask1_1d > 0) | (mask2_1d > 0)
    fg2 = (mask1_flip_x_1d > 0) | (mask2_1d > 0)
    fg3 = (mask1_1d > 0) | (mask2_flip_y_1d > 0)
    fg4 = (mask1_flip_x_1d > 0) | (mask2_flip_y_1d > 0)
    # Scores.
    score1 = normalized_mutual_info_score(label1_1d[fg1], label2_1d[fg1])
    score2 = normalized_mutual_info_score(label1_flip_x_1d[fg2], label2_1d[fg2])
    score3 = normalized_mutual_info_score(label1_1d[fg3], label2_flip_y_1d[fg3])
    score4 = normalized_mutual_info_score(label1_flip_x_1d[fg4], label2_flip_y_1d[fg4])
    return score1, score2, score3, score4


def local_gmm_compare(label1: np.ndarray, label2: np.ndarray,
                      means1: pd.Series, means2: pd.Series):
    """
    :param label1: labeled image, 2D numpy array
    :param label2: labeled image, 2D numpy array
    :param means1: means for points in different levels
    :param means2: means for points in different levels
    :return: local gmm score
    Compare 2 masks in 4 orientations and return the best score.
    """
    label1_flip_x = np.flip(label1, axis=0)
    label2_flip_y = np.flip(label2, axis=1)
    score1 = local_score(label1, label2, means1, means2)
    score2 = local_score(label1_flip_x, label2, means1, means2)
    score3 = local_score(label1, label2_flip_y, means1, means2)
    score4 = local_score(label1_flip_x, label2_flip_y, means1, means2)
    return score1, score2, score3, score4
