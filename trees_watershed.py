import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def solution1(img):
    for i in range(0,10):
        for j in range(0,10):
            k = img[i,j]
            print(k)
            ax.scatter(i, j, k, color='black')

def scatter_plot(img):
    img = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
    #img_blurred = gaussian_filter(img, (3,3))
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(xx, yy, img[:,:,0], rstride=1, cstride=1, cmap=plt.cm.gray,
                linewidth=0)

    img = img.astype('float32')/255
    ax.plot_surface(xx, yy, np.atleast_2d(0) , rstride=1, cstride=1, facecolors=img)
    plt.show()


def rgb_scatter_plot(img):
    r, g, b = cv2.split(img)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    pix_col = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))


if __name__ == "__main__":
    frame = cv2.imread("doc/trees_rgb.jpg", 1, )
    #frame = frame[0:1000, 0:1000]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 713, 9)

    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow("thresh", thresh)

    kernel = np.ones((9,9), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
            iterations=3)

    cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
    cv2.imshow("opening", opening)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    cv2.namedWindow("sure_bg", cv2.WINDOW_NORMAL)
    cv2.imshow("sure_bg", sure_bg)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform,
            0.05*dist_transform.max(), 255, 0)


    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    cv2.namedWindow("sure_fg", cv2.WINDOW_NORMAL)
    cv2.imshow("sure_fg", sure_fg)

    markers = markers + 1
    markers[unknown==255] = 0

    markers = cv2.watershed(frame, markers)
    frame[markers==-1] = [0, 255, 0]

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)


    green = frame[:,:,2]



    cv2.waitKey(0)
    cv2.destroyAllWindows()
