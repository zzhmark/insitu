import numpy as np

import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte
from skimage.measure import block_reduce
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from sklearn.preprocessing import normalize

from utils import Step


def std_dev_filter(image, kernel):
    """
    :param image: grayscale image, 2D numpy array
    :param kernel: 2-element tuple
    :return: 2D numpy float array
    Calculate standard deviation on an array, based on a specific kernel.
    """
    a = image.astype(np.float)
    return cv2.sqrt(cv2.blur(a ** 2, kernel) - cv2.blur(a, kernel) ** 2)


def fillHole(mask):
    """
    :param mask: grayscale image, 2D numpy array
    :return: binary image, 2D numpy array
    Find contours in an array image, fill the out most one.
    """
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(mask, contour, -1, 255, cv2.FILLED)


def extract(image, kernel, thr):
    """
    :param image: 3D numpy array
    :param kernel: 2-element tuple
    :param thr: minimum sd
    :return: (3D numpy array, 2D numpy array)
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_sd = std_dev_filter(image_gray, kernel)
    image_thr = cv2.threshold(image_sd, thr, 255,
                              cv2.THRESH_BINARY)[1].astype(np.uint8)
    mask_out = fillHole(image_thr)
    mask_color = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    image_out = cv2.bitwise_or(image, cv2.bitwise_not(mask_color))
    return image_out, mask_out


def fgPts(mask):
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


def angle_of_vectors(v1, v2):
    """
    :param v1: 2D vector
    :param v2: 2D vector
    :return: the angle of v1 and v2 in degrees
    """
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot / norm))


def rotate(image, mask):
    """
    :param image: 3D numpy array
    :param mask: 2D binary numpy array
    :return: (3D numpy array, 2D numpy array)
    """
    # Make x, y axis correspond to the width, height direction.
    pts = np.flip(fgPts(mask))
    # PCA to determine the main direction and construct the transform matrix.
    pca = PCA(2)
    pca.fit(pts)
    angle = angle_of_vectors(pca.components_[0, :], (1, 0))
    affine_mat = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    # Get transformed foreground points to find their x, y limit for translating.
    new_pts = np.dot(affine_mat[:, :2], pts.transpose())
    x_min, x_max = new_pts[0, :].min(), new_pts[0, :].max()
    y_min, y_max = new_pts[1, :].min(), new_pts[1, :].max()
    width, height = int(x_max - x_min), int(y_max - y_min)
    affine_mat[:, 2] = [-x_min, -y_min]
    # Apply transformation.
    image_out = cv2.warpAffine(image, affine_mat, (width, height),
                               borderValue=(255, 255, 255))
    mask_out = cv2.warpAffine(mask, affine_mat, (width, height),
                              borderValue=0)
    return image_out, mask_out


def register(image, mask, dsize):
    """
    :param image: 3D numpy array
    :param mask: 2D numpy array
    :param dsize: (width, height)
    :return: (3D numpy array, 2D numpy array)
    """
    image_rot, mask_rot = rotate(image, mask)
    image_out = cv2.resize(image_rot, dsize)
    mask_out = cv2.resize(mask_rot, dsize)
    return image_out, mask_out


def gmm(data, n, method='default'):
    """
    :param data: list of input
    :param n: number of components
    :param method: 'default' or 'bayesian'
    :return: (labels, means)
    """
    # To avoid error, the number of components should be
    # no more than the length of input data.
    noc = min(len(data), n)
    if method.lower() == 'bayesian':
        model = BayesianGaussianMixture(n_components=noc)
        model.fit(data)
    else:
        model = GaussianMixture(n_components=noc)
        model.fit(data)
    return model.predict(data), model.means_


def sigmoid(x, cutoff, gain):
    """
    :param x:
    :param cutoff:
    :param gain:
    :return:
    """
    return 1 / (1 + np.exp(gain * (cutoff - x)))


def adjust2gray(image):
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
    # b = image[:, :, 0].astype(np.float)
    # g = image[:, :, 1].astype(np.float)
    # r = image[:, :, 2].astype(np.float)
    # out = b - (g + r) / 2
    # out[out > 255] = 255
    # out[out < 0] = 0
    # image_out = 255 - out.astype(np.uint8)
    return image_out


def global_gmm(image, mask, n, patch):
    """
    :param image: color image, 3D numpy array
    :param mask: binary image, 2D binary numpy array
    :param n: number of components
    :param patch: size of block
    :return: (3D numpy array, 2D numpy array, pandas data frame)
    Solve GMM for the image histogram, iteratively find
    the minimalist mean of GMM models and separate the
    corresponding points.
    """
    # Convert color image to grayscale.
    # Strengthen staining signals and remove false positive patterns.
    image_adjusted = adjust2gray(image)
    # Down sample the images and masks to reduce calculation.
    image_down = block_reduce(image_adjusted, patch,
                              np.mean, 255).astype(np.uint8)
    mask_down = block_reduce(mask, patch, np.min)
    global_mean = int(np.mean(image_down[mask_down > 0]))
    mask_out = np.zeros(mask_down.shape, dtype=np.uint8)
    nol = 0  # The count of labels.
    model_out = pd.DataFrame(columns=[0, 1])
    while True:
        # Retrieve current foreground points.
        pts = fgPts(mask_down)
        # The model fits the foreground pixels' intensity
        model_input = np.array([[image_down[tuple(p)]
                                 for p in pts]]).transpose()
        labels, means = gmm(model_input, n)
        min_cluster, min_mean = np.argmin(means), min(means)
        # When the minimum mean reach the global mean, break the loop.
        if min_mean >= global_mean:
            break
        # Otherwise, label the points in the output mask,
        # and dump them in the next run.
        stain_pts = pts[labels == min_cluster]
        nol += 1
        for p in stain_pts:  #
            mask_down[tuple(p)] = 0
            mask_out[tuple(p)] = nol
        model_out = model_out.append([[nol, min_mean]])
    model_out.columns = ['label', 'mean']
    model_out = model_out.set_index('label')
    # Label the output image.
    height, width = image.shape[:2]
    mask_up = cv2.resize(mask_out, (width, height),
                         interpolation=cv2.INTER_NEAREST)
    image_out = img_as_ubyte(label2rgb(mask_up, image, bg_label=0))
    return image_out, mask_out, model_out


def local_gmm(image, mask, global_model, n):
    """
    :param image: bgr image, 3D numpy array
    :param mask: binary image, 2D numpy array
    :param global_model: parameters of global gmm, pandas data frame
    :param n: maximum number of components for the bayesian algorithm
    :return: (image, mask, model)
    Solve GMM for points' local distribution within
    each grayscale level generated by the global model.
    """
    mask_out = np.zeros(mask.shape, dtype=np.uint8)
    model_out = pd.DataFrame(columns=['label', 'mean'])
    # Iterate over different grayscale levels in the global model.
    for i, mean in zip(global_model.index, global_model['mean']):
        pts = fgPts(mask == i)  # Retrieve points with a specific label.
        labels = gmm(pts, n, 'bayesian')[0]
        # Adjust labels from 0..n-1 to 1..n.
        # Because labels can be discontinuous.
        levels = np.unique(labels)
        labels = [np.where(levels == i)[0][0] + 1 for i in labels]
        # Label the areas on the output mask.
        start = np.max(mask_out)
        for p, label in zip(pts, labels):
            mask_out[tuple(p)] = start + label
        model = pd.DataFrame({'label': [*range(start, start + max(labels))],
                              'mean': [mean] * max(labels)})
        model_out = model_out.append(model)
    model_out = model_out.set_index('label')
    # Label the output image
    height, width = image.shape[:2]
    mask_up = cv2.resize(mask_out, (width, height),
                         interpolation=cv2.INTER_NEAREST)
    image_out = img_as_ubyte(label2rgb(mask_up, image, bg_label=0))
    return image_out, mask_out, model_out


def blob_score(pts1, pts2, mean1, mean2):
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


def local_score(mask1, mask2, means1, means2):
    """
    :param mask1: labeled image, 2D numpy array
    :param mask2: labeled image, 2D numpy array
    :param means1: means for points in different levels
    :param means2: means for points in different levels
    :return: the local score
    Calculate 2 images' similarity score, by working out similarity
    scores between any 2 blobs, and sum up the best matches.
    """
    n1, n2 = len(means1), len(means2)
    # List points for different levels.
    pts_list1 = [fgPts(mask1 == i + 1) for i in range(n1)]
    pts_list2 = [fgPts(mask2 == i + 1) for i in range(n2)]
    score_blobs = np.zeros((n1, n2))
    for i, pts1, mean1 in zip(range(n1), pts_list1, means1):
        for j, pts2, mean2 in zip(range(n2), pts_list2, means2):
            score_blobs[i, j] = blob_score(pts1, pts2, mean1, mean2)
    return np.max(score_blobs, axis=0).sum() + np.max(score_blobs, axis=1).sum()


def batch_apply(step, **kwargs):
    """
    :param step: enum for steps
    :param kwargs: key word arguments
    :return: depending on step
    """
    images = {}
    masks = {}

    if step == Step.EXTRACT:
        for key, image in kwargs['images'].items():
            images[key], masks[key] = \
                extract(image, kwargs['kernel'], kwargs['thr'])
        return images, masks

    if step == Step.REGISTER:
        for key, image, mask in zip(kwargs['keys'],
                                    kwargs['images'],
                                    kwargs['masks']):
            images[key], masks[key] = \
                register(image, mask, kwargs['size'])
        return images, masks

    gmm_models = {}

    if step == Step.GLOBAL_GMM:
        for key, image, mask in zip(kwargs['keys'],
                                    kwargs['images'],
                                    kwargs['masks']):
            images[key], masks[key], gmm_models[key] = \
                global_gmm(image, mask, kwargs['nok'], kwargs['patch'])
        return images, masks, gmm_models

    if step == Step.LOCAL_GMM:
        for key, image, mask, model in zip(kwargs['keys'],
                                           kwargs['images'],
                                           kwargs['masks'],
                                           kwargs['global_models']):
            images[key], masks[key], gmm_models[key] = \
                local_gmm(image, mask, model, kwargs['nok'])
        return images, masks, gmm_models

    if step == Step.SCORE:
        n = len(kwargs['keys'])

        # global
        global_score_table = np.zeros((n, n))
        for i, mask1 in zip(range(n), kwargs['global_masks'].values()):
            for j, mask2 in zip(range(n), kwargs['global_masks'].values()):
                mask1_flip_x = np.flip(mask1, axis=0)
                mask2_flip_y = np.flip(mask2, axis=1)
                mask1_1d = np.resize(mask1, -1)
                mask2_1d = np.resize(mask2, -1)
                mask1_flip_x_1d = np.resize(mask1_flip_x, -1)
                mask2_flip_y_1d = np.resize(mask2_flip_y, -1)
                global_score_table[i, j] = \
                    max(normalized_mutual_info_score(mask1_1d, mask2_1d),
                        normalized_mutual_info_score(mask1_flip_x_1d, mask2_1d),
                        normalized_mutual_info_score(mask1_1d, mask2_flip_y_1d),
                        normalized_mutual_info_score(mask1_flip_x_1d, mask2_flip_y_1d))

        # local
        local_score_table = np.zeros((n, n))
        for i, mask1, model1 in zip(range(n),
                                    kwargs['local_masks'].values(),
                                    kwargs['global_models'].values()):
            for j, mask2, model2 in zip(range(n),
                                        kwargs['local_masks'].values(),
                                        kwargs['global_models'].values()):
                mask1_flip_x = np.flip(mask1, axis=0)
                mask2_flip_y = np.flip(mask2, axis=1)
                means1 = model1['mean']
                means2 = model2['mean']
                local_score_table[i, j] = \
                    max(local_score(mask1, mask2, means1, means2),
                        local_score(mask1_flip_x, mask2, means1, means2),
                        local_score(mask1, mask2_flip_y, means1, means2),
                        local_score(mask1_flip_x, mask2_flip_y, means1, means2))
        local_score_table = normalize(local_score_table, norm='max', axis=0)

        global_score_table = pd.DataFrame(global_score_table,
                                          columns=kwargs['keys'],
                                          index=kwargs['keys'])
        local_score_table = pd.DataFrame(local_score_table,
                                         columns=kwargs['keys'],
                                         index=kwargs['keys'])
        # hybrid
        hybrid_score_table = global_score_table * local_score_table
        return global_score_table, local_score_table, hybrid_score_table
