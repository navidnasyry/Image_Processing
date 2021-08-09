

import argparse
import cv2
import numpy as np



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help= "path to input image")

args = vars(ap.parse_args())



image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))

#blackhat operation which enables us to find dark regions on a light backtround
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)


#tophat/whitehat will enable us to find light regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)




cv2.imshow("original", image)
cv2.waitKey(0)


cv2.imshow("gray", gray)
cv2.waitKey(0)


cv2.imshow("blackhat", blackhat)
cv2.waitKey(0)


cv2.imshow("tophat", tophat)
cv2.waitKey(0)


cv2.destroyAllWindows()
cv2.imshow("original", image)


kernelSize = [(3,3)]
for kernel_size in kernelSize:
    #gradiant
    #A morphological gradient is the difference between a dilation and erosion. 
    #It is useful for determining the outline of a particular object of an image:

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("openning: ({} , {})".format(
        kernel_size[0], kernel_size[1]), gradient
    )    

    cv2.waitKey(0)


#binary
(thresh, im_bw) = cv2.threshold(gradient, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("binary", im_bw)
cv2.waitKey(0)


thresh = 110
im_bw = cv2.threshold(gradient, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("binary", im_bw)
cv2.waitKey(0)



print(im_bw)
indices = np.argwhere(im_bw == [0,0,0])
print(indices)
coordinates = zip(indices[0], indices[1])
print(coordinates)
cv2.line(image, coordinates, 4)



cv2.imshow("line", image)
cv2.waitKey(0)
