import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte
from skimage.measure import block_reduce
import math
from sklearn.metrics import normalized_mutual_info_score
from utils import Step
import pandas as pd

def sd(arr, kernel):
    '''
    :param arr: 2D numpy array
    :param kernel: 2-element tuple
    :return: 2D numpy float array
    '''
    a = arr.astype(np.float32)
    return cv2.sqrt(cv2.blur(a ** 2, kernel) - cv2.blur(a, kernel) ** 2)

def fillHole(arr):
    '''
    :param arr: 2D binary numpy array
    :return: 2D binary numpy array
    '''
    contour = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(arr, contour, -1, 255, cv2.FILLED)

def crop(arr, mask):
    '''
    :param arr: 3D numpy array
    :param mask: 2D binary numpy array
    :return: 3D numpy array
    '''
    img = 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255 * (255 - arr)
    return img.astype(np.uint8)

def extract(img, kernel, thr):
    '''
    :param img: 3D numpy array
    :param kernel: 2-element tuple
    :param thr: minimum sd
    :return: (3D numpy array, 2D numpy array)
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sd = sd(img_gray, kernel)
    img_sd_thr = cv2.threshold(img_sd, thr, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    img_sd_fill = fillHole(img_sd_thr)
    img_out = crop(img, img_sd_fill)
    return img_out, img_sd_fill

def fgPts(arr):
    '''
    :param arr: 2D binary numpy array
    :return: 2 * n numpy matrix
    '''
    height, width = arr.shape
    return np.array([[i, j] for i in range(height) for j in range(width) if arr[i, j] > 0])

def angle_of_vectors(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mod = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]) * math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    return math.degrees(math.acos(dot / mod))

def rotate(img, mask):
    '''
    :param img: 3D numpy array
    :param mask: 2D binary numpy array
    :return: (3D numpy array, 2D numpy array)
    '''
    pts = np.flip(fgPts(mask))
    pca = PCA(2).fit(pts)
    angle = angle_of_vectors(pca.components_[0, :], (1, 0))
    affineMat = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    new_pts = np.dot(affineMat[:, :2], pts.transpose())
    x_min, x_max, y_min, y_max = new_pts[0, :].min(), new_pts[0, :].max(), new_pts[1, :].min(), new_pts[1, :].max()
    width, height = int(x_max - x_min), int(y_max - y_min)
    affineMat[:, 2] = [-x_min, -y_min]
    img_out = cv2.warpAffine(img, affineMat, (width, height), borderValue=(255, 255, 255))
    mask_out = cv2.warpAffine(mask, affineMat, (width, height), borderValue=0)
    return img_out, mask_out

def register(img, mask, dsize):
    '''
    :param img: 3D numpy array
    :param mask: 2D numpy array
    :param dsize: (width, height)
    :return: (3D numpy array, 2D numpy array)
    '''
    img_rot, mask_rot = rotate(img, mask)
    img_out, mask_out = cv2.resize(img_rot, dsize), cv2.resize(mask_rot, dsize)
    return img_out, mask_out

def gmm_hist(img, pts, n):
    '''
    :param img: 3D numpy array
    :param pts: list of foreground points
    :param n: number of clusters
    :return: (label, means)
    '''
    hist = [img[p[0], p[1]] for p in pts]
    gm = GaussianMixture(n).fit(hist)
    lab = gm.predict(hist)
    means = [np.mean(gm.means_[i,:]) for i in range(n)]
    return lab, means

def global_gmm(img, mask, n, patch):
    '''
    :param img: 3D numpy array
    :param mask: 2D binary numpy array
    :param n: number of clusters
    :param block: size of block
    :return: (3D numpy array, 2D numpy array)
    '''
    img_down = block_reduce(img, (*patch, 3), np.mean)
    mask_down = block_reduce(mask, patch, np.min)
    all_mean = np.mean(img_down[mask_down > 0, :])
    mask_out = np.zeros(mask_down.shape, dtype=np.uint8)
    nLabel = 0
    mean_set_out = []
    while True:
        pts = fgPts(mask_down)
        label, means = gmm_hist(img_down, pts, n)
        min_cluster, min_mean = np.argmin(means), min(means)
        if min_mean >= all_mean:
            break
        mean_set_out.append(min_mean)
        stain_pts = pts[label == min_cluster]
        nLabel += 1
        for p in stain_pts:
            mask_down[tuple(p)] = 0
            mask_out[tuple(p)] = nLabel
    height, width = mask.shape
    mask_up = cv2.resize(mask_out, (width, height), interpolation=cv2.INTER_NEAREST)
    img_out = img_as_ubyte(label2rgb(mask_up, img, bg_label=0))
    return img_out, mask_out, mean_set_out

def gmm_blob(pts, n):
    '''
    :param pts:
    :param n:
    :return:
    '''
    gm = BayesianGaussianMixture(n_components=min(len(pts), n)).fit(pts)
    lab = gm.predict(pts)
    return lab

def local_gmm(img, mask, mean_set, n):
    '''
    :param img:
    :param mask:
    :param mean_set:
    :param n:
    :return:
    '''
    mask_out = np.zeros(mask.shape, dtype=np.uint8)
    mean_set_out = []
    for i, mean in zip(range(mask.max()), mean_set):
        pts = fgPts(mask == i + 1)
        label = gmm_blob(pts, n)
        level = np.unique(label)
        label = [np.where(level == i)[0][0] + 1 for i in label]
        start = mask_out.max()
        for p, lab in zip(pts, label):
            mask_out[tuple(p)] = start + lab
        mean_set_out.extend([mean] * np.max(label))
    height, width = img.shape[:2]
    mask_up = cv2.resize(mask_out, (width, height), interpolation=cv2.INTER_NEAREST)
    img_out = img_as_ubyte(label2rgb(mask_up, img, bg_label=0))
    return img_out, mask_out, mean_set_out

def blob_score(pts1, pts2, mean1, mean2):
    '''
    :param mask1:
    :param mask2:
    :param mean1:
    :param mean2:
    :return:
    '''
    grad_term = 1 - np.abs(mean1 - mean2) / 256
    set1, set2 = set(tuple(i) for i in pts1), set(tuple(i) for i in pts2)
    overlap_term = len(set1.intersection(set2)) / len(set2.union(set2))
    return grad_term * overlap_term

def local_score(mask1, mask2, mean_set1, mean_set2):
    '''
    :param mask1:
    :param mask2:
    :param mean_set1:
    :param mean_set2:
    :return:
    '''
    n1, n2 = len(mean_set1), len(mean_set2)
    pts1, pts2 = [fgPts(mask1 == i + 1) for i in range(n1)], \
                 [fgPts(mask2 == i + 1) for i in range(n2)]
    score_blobs = np.zeros((n1, n2))
    for i, p1, mean1 in zip(range(n1), pts1, mean_set1):
        for j, p2, mean2 in zip(range(n2), pts2, mean_set2):
            score_blobs[i, j] = blob_score(p1, p2, mean1, mean2)
    return score_blobs.max(axis=0).sum() + score_blobs.max(axis=1).sum()

def batch_apply(step, **kwargs):
    '''
    :param step: string
    :param kwargs:
    :return:
    '''
    images = {}
    masks = {}

    if step == Step.EXTRACT:
        for key, img in kwargs['images'].items():
            images[key], masks[key] = extract(img, kwargs['kernel'], kwargs['thr'])
        return images, masks

    if step == Step.REGISTER:
        for key, img, mask in zip(kwargs['keys'], 
                                  kwargs['images'], 
                                  kwargs['masks']):
            images[key], masks[key] = register(img, mask, kwargs['size'])
        return images, masks

    mean_sets = {}

    if step == Step.GLOBAL_GMM:
        for key, img, mask in zip(kwargs['keys'], 
                                  kwargs['images'], 
                                  kwargs['masks']):
            images[key], masks[key], mean_sets[key] = global_gmm(img, mask, kwargs['nok'], kwargs['patch'])
        return images, masks, mean_sets

    if step == Step.LOCAL_GMM:
        for key, img, mask, mean_set in zip(kwargs['keys'], 
                                            kwargs['images'], 
                                            kwargs['masks'], 
                                            kwargs['mean_sets']):
            images[key], masks[key], mean_sets[key] = local_gmm(img, mask, mean_set, kwargs['nok'])
        return images, masks, mean_sets

    if step == Step.SCORE:
        n = len(kwargs['keys'])
        
        # global
        global_score_table = np.zeros((n, n))
        mask_batch_1d = [np.reshape(mask, -1) for mask in kwargs['global_masks'].values()]
        for i, mask1 in zip(range(n), mask_batch_1d):
            for j, mask2 in zip(range(n), mask_batch_1d):
                global_score_table[i, j] = normalized_mutual_info_score(mask1, mask2)
                
        # local
        local_score_table = np.zeros((n, n))
        for i, mask1, mean_set1 in zip(range(n),
                                    kwargs['local_masks'].values(),
                                    kwargs['mean_sets'].values()):
            for j, mask2, mean_set2 in zip(range(n),
                                        kwargs['local_masks'].values(),
                                        kwargs['mean_sets'].values()):
                local_score_table[i, j] = local_score(mask1, mask2, mean_set1, mean_set2)

        global_score_table = pd.DataFrame(global_score_table,
                                          columns=kwargs['keys'],
                                          index=kwargs['keys'])
        local_score_table = pd.DataFrame(local_score_table,
                                         columns=kwargs['keys'],
                                         index=kwargs['keys'])
        hybrid_score_table = global_score_table * local_score_table
        return global_score_table, local_score_table, hybrid_score_table
