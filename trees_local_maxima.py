"""Forest tree crowns counting


Author: Artyom Voronin
Brno, 2021
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from skimage.feature import peak_local_max


def plot2d(px, py, img_raw, img_ret):
    """Plot 2d image with pinpoint
    markers on possible tree crown
    and save to "tree_detected.jpg" file.
    """
    for i in range(len(px)):
        cv2.drawMarker(img_raw, (px[i], py[i]), (0, 0, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
                line_type=cv2.LINE_AA)
    cv2.namedWindow("Detected trees crowns", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected trees crowns", img_raw)
    cv2.imwrite('doc/trees_detected.jpg', img_raw)

def local_maximum(img, plot=False):
    """Local maximum filtering

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gaussian_filter(gray, (8,8))

    xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    zz = gray[:,:]

    peaks = peak_local_max(zz, min_distance=30, threshold_abs=100)
    pp = np.asarray(peaks)
    px, py = pp[:,1], pp[:,0]
    pz = zz[px, py]

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = 500
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, (size,size), interpolation=cv2.INTER_LINEAR)

        gray = gaussian_filter(gray, (10,10))

        xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        zz = gray[:,:]

        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                cmap='viridis', linewidth=0)

        img = img.astype('float32')/255
        ax.plot_surface(xx, yy, np.atleast_2d(0) , rstride=1, cstride=1, facecolors=img)
    return px, py, pz

def segmentation(img, crop=False):
    if crop:
        print("Cropping image")
        img = img[0:1000, 0:1000]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    ret = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)

    return ret, gray, img


if __name__ == "__main__":
    frame = cv2.imread("doc/trees_rgb.jpg", 1) # Path to image

    ret, _, frame = segmentation(frame, crop=False)
    px, py, pz = local_maximum(ret, plot=False)
    print(f"Number of detected tree crowns: {len(px)}")
    plot2d(px, py, frame, ret)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
