import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

def Segment_Otsu(FileIm):
    img = cv2.imread(FileIm)
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(thresh, 'gray')
    cv2.imwrite("output_data\segmentOTSU.jpg", thresh)
    plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return

def Segment_Robert(FileIm):
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])
    img = cv2.imread(FileIm, 0).astype('float64')
    img /= 255.0
    img_rbg = cv2.imread(FileIm)
    b, g, r = cv2.split(img_rbg)
    rgb_img = cv2.merge([r, g, b])
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite("output_data\segmentRobert.jpg", edged_img)
    plt.title("Robert operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return

def Segment_kmeans(FileIm):
    img = cv2.imread(FileIm)
    img_rbg = cv2.imread(FileIm)
    b, g, r = cv2.split(img_rbg)
    rgb_img = cv2.merge([r, g, b])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts = 10
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    edged_img = res.reshape((img.shape))
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite("output_data\segmentKmeans.jpg", edged_img)
    plt.title("kmeans operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return

Segment_Otsu(r'input_data\etalon.jpg')
Segment_Otsu(r'input_data\operativniy.jpg')
Segment_Robert('input_data\etalon.jpg')
Segment_Robert('input_data\operativniy.jpg')
Segment_kmeans('input_data\etalon.jpg')
Segment_kmeans('input_data\operativniy.jpg')