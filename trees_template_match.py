import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    frame = cv2.imread("doc/trees_rgb.jpg", 1)
    #frame = frame[0:1000, 0:1000]
    template = cv2.imread("doc/tm_tree1.png", 1)

    print(template.shape[::-1])
    _, w, h = template.shape[::-1]

    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.1
    loc = np.where(res>=threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0]+w, pt[1]+h), (0,0,255), 2)


    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

