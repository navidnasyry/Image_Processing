

import cv2 as cv
import numpy as np
import argparse


def showImage(name, img):

    #cv2.imshow("original", image)
    winname = name
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv.imshow(winname, img)
    pic_name = "image_"+ name +".png"
    cv.imwrite(pic_name, img)
    cv.waitKey(0)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help= "path to input image")

args = vars(ap.parse_args())

img = cv.imread(args['image'])

showImage("base", img)


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)
kernel = np.ones((2,2), np.uint8)
print(kernel)


dilation = cv.dilate(thresh, kernel, iterations=2)
erotin = cv.erode(thresh, kernel,iterations=18 )


contours_dilation, hierarchy = cv.findContours(dilation,
                                    cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)


contours_erotion, hierarchy2 = cv.findContours(erotin,
                                    cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)

print(contours_dilation)
print(contours_erotion)


objects_erotion = str(len(contours_erotion))
msg = "Number of Objects in Erotion : " + objects_erotion
print(msg)
cv.putText(erotin, msg, (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.4,(240,0,150), 1)


objects_dilation = str(len(contours_dilation))
msg = "Number of Objects in Dilation: " + objects_dilation
print(msg)
cv.putText(dilation, msg, (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.4,(240,0,150), 1)


showImage('thresh', thresh)
showImage('original', img)
showImage('Thresh', thresh)
showImage('Dilation', dilation)
showImage("Erotion", erotin)

cv.destroyAllWindows()


















