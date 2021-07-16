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

def solution2(img):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    img = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
    img_blurred = gaussian_filter(img, (5,5))
    img = img_blurred
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray,
                linewidth=0)

    #zz = np.zeros(xx.shape)
    #img = img.astype('float32')/255
    #ax.plot_surface(xx, yy, np.atleast_2d(0) , rstride=1, cstride=1, facecolors=img)
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("trees_rgb.jpg", 1, )
    img = img[0:1000, 4000:5000]
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img[1][1])

    cv2.imshow('trees', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
