import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte
from skimage.measure import block_reduce
import math

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

def registrate(img, mask, dsize):
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
    while True:
        pts = fgPts(mask_down)
        label, means = gmm_hist(img_down, pts, n)
        min_cluster, min_mean = np.argmin(means), min(means)
        if min_mean >= all_mean:
            break
        stain_pts = pts[label == min_cluster]
        nLabel += 1
        for p in stain_pts:
            mask_down[p[0], p[1]] = 0
            mask_out[p[0], p[1]] = nLabel
    height, width = mask.shape
    mask_up = cv2.resize(mask_out, (width, height), interpolation=cv2.INTER_NEAREST)
    img_out = img_as_ubyte(label2rgb(mask_up, img, bg_label=0))
    return img_out, mask_out

def gmm_blob(pts, n):
    '''
    :param pts:
    :param n:
    :return:
    '''
    gm = BayesianGaussianMixture(n_components=n).fit(pts)
    lab = gm.predict(pts)
    return lab

def local_gmm(img, mask, n):
    '''
    :param mask:
    :param n:
    :return:
    '''
    mask_sum = np.zeros(mask.shape, dtype=np.uint8)
    mask_out = []
    count = 1
    for level in range(1, mask.max() + 1):
        pts = fgPts(mask == level)
        label = gmm_blob(pts, min(len(pts), n))
        mask_out.append(label)
        for i in range(len(pts)):
            mask_sum[pts[i][0], pts[i][1]] = count + label[i]
        count += label.max()
    height, width = img.shape[:2]
    mask_up = cv2.resize(mask_sum, (width, height), interpolation=cv2.INTER_NEAREST)
    img_out = img_as_ubyte(label2rgb(mask_up, img, bg_label=0))
    return img_out, mask_out