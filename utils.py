from enum import Enum

from PyQt5.QtGui import QImage, QPixmap
import cv2


class Step(Enum):
    RAW = 0
    EXTRACT = 1
    REGISTER = 2
    GLOBAL_GMM = 3
    LOCAL_GMM = 4
    HYBRID = 5


def cvimg2qpixmap(img):
    """
    :param img: numpy array, bgr or grayscale
    :return: QPixmap
    """
    if len(img.shape) == 2:
        height, width = img.shape
        return QPixmap(QImage(img, width, height, width, QImage.Format_Grayscale8))
    else:
        height, width, channel = img.shape
        return QPixmap(QImage(img, width, height, width * channel, QImage.Format_RGB888).rgbSwapped())


def fitView(arr, view):
    """
    :param arr: 2D or 3D numpy array
    :param view: QGraphicsView
    :return: 2D or 3D numpy array
    """
    img_h, img_w = arr.shape[:2]
    view_size = view.geometry().size()
    view_h, view_w = view_size.height() - 2, view_size.width() - 2
    scale = min(view_h / img_h, view_w / img_w)
    size = int(img_w * scale), int(img_h * scale)
    return cv2.resize(arr, size)
