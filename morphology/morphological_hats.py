

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




#cv2.imshow("original", image)
winname = "original"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, image)
cv2.waitKey(0)


#cv2.imshow("gray", gray)
winname = "gray"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, gray)
cv2.waitKey(0)


#cv2.imshow("blackhat", blackhat)
winname = "blackhat"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, blackhat)
cv2.waitKey(0)


#cv2.imshow("tophat", tophat)
winname = "tophat"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, tophat)
cv2.waitKey(0)


cv2.destroyAllWindows()
#cv2.imshow("original", image)
winname = "original"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, image)
cv2.waitKey(0)

kernelSize = [(3,3)]
for kernel_size in kernelSize:
    #gradiant
    #A morphological gradient is the difference between a dilation and erosion. 
    #It is useful for determining the outline of a particular object of an image:

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("openning: ({} , {})".format(
    #     kernel_size[0], kernel_size[1]), gradient
    #     )    
    winname = "openning: ({} , {})".format(
        kernel_size[0], kernel_size[1])
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, gradient)
    cv2.waitKey(0)



#binary
(thresh, im_bw) = cv2.threshold(gradient, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#cv2.imshow("binary", im_bw)
winname = "binary"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, im_bw)
cv2.waitKey(0)


thresh = 110
im_bw = cv2.threshold(gradient, thresh, 255, cv2.THRESH_BINARY)[1]

#cv2.imshow("binary", im_bw)
winname = "binary"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, im_bw)
cv2.waitKey(0)


kernelSize = [(2,1)]

for kernel_size in kernelSize:
    #opening
    #first an erosion is applied to remove the small blobs,
    #  then a dilation is applied to regrow the size of the original object.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
    winname = "binary"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, im_bw)
    cv2.waitKey(0)






print(im_bw)
print(im_bw.ndim)
print(im_bw.shape)
print(type(im_bw))
indices = np.argwhere(im_bw == [255])

print(indices.ndim)
print(indices.shape)

print(indices)
# coordinates = zip(indices[0], indices[1])
# print()
# print(list(coordinates))
print()
coordinates_tuples = list(map(tuple, indices))
coordinates_tuples = [x[::-1] for x in coordinates_tuples]
print(coordinates_tuples)
coordinates_tuples = np.array(coordinates_tuples)
print()
print(coordinates_tuples)
print(type(coordinates_tuples))
#cv2.line(image, coordinates, 4)
# for point1, point2 in zip(coordinates[0], coordinates[1]): 
#     cv2.line(image, point1, point2, [0, 255, 0], 2) 
cv2.drawContours(image, [coordinates_tuples], -1, (0,0,255), 2)
#cv2.drawContours(im_bw, [coordinates_tuples], 0, (0,0,255), 2)


#cv2.imshow("line", image)
winname = "line1"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, image)
cv2.waitKey(0)


#cv2.imshow("before", im_bw)
winname = "before"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, im_bw)
cv2.waitKey(0)


