import numpy as np
import cv2 as cv
import argparse

import sys


def showImage(name, img):

    #cv2.imshow("original", image)
    winname = name
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv.imshow(winname, img)
    pic_name = "image_"+ name +".png"
    cv.imwrite(pic_name, img)
    cv.waitKey(0)


def main():
  
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, 
                    help= "path to input image")

    args = vars(ap.parse_args())

    src = cv.imread(args['image'])

    showImage("base", src)

    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #showImage('gray', img)

    img = cv.medianBlur(img, 5)

    showImage('blur', img)

    cimg = src.copy() # numpy function

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 69, 21, 2, 26)

    print(circles)

    counter = 0
    if circles is not None: # Check if circles have been found and only then iterate over these and add them to the image
        _a, b, _c = circles.shape
        for i in range(b):
            cv.circle(cimg, (int(circles[0][i][0]), int(circles[0][i][1])), int(circles[0][i][2]), (0, 0, 255), 2, cv.LINE_AA)
            cv.circle(cimg, (int(circles[0][i][0]), int(circles[0][i][1])), 2, (0, 255, 0), 3, cv.LINE_AA)  # draw center of circle
            counter += 1

        print(f'counter ;', counter)
        showImage("detected circles", cimg)

    showImage("source", src)
    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()

