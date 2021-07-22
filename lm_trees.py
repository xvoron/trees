import numpy as np
import cv2
import matplotlib.pyplot as plt

def scatter_plot(img):
    img = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(xx, yy, img[:,:], rstride=1, cstride=1, cmap=plt.cm.gray,
                linewidth=0)

    #img = img.astype('float32')/255
    #ax.plot_surface(xx, yy, np.atleast_2d(0) , rstride=1, cstride=1, facecolors=img)
    plt.show()

if __name__ == "__main__":
    frame = cv2.imread("doc/trees_rgb.jpg", 1)
    #frame = frame[0:500, 0:500]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])


    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow("res", res)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.dilate(gray, (5,5), iterations=3)
    gray = cv2.GaussianBlur(gray, (3,3), 0)


    #scatter_plot(gray)
    contours, hier = cv2.findContours(cv2.bitwise_not(gray), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('found', len(contours), 'contours')
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.imshow("gray", gray)

    #blur = cv2.GaussianBlur(res, (5,5), 0)

    #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.imshow("frame", frame)

    #cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
    #cv2.imshow("blur", blur)

    #cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    #cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

