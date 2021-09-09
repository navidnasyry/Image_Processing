 

import cv2
import argparse



def showImage(img, name):

    #cv2.imshow("original", image)
    winname = name
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    pic_name = "image_"+ name +".png"
    cv2.imwrite(pic_name, img)
    cv2.waitKey(0)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help= "path to input image")

args = vars(ap.parse_args())



image = cv2.imread(args['image'])
showImage(image, "original")



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showImage(gray, "gray")



ret, thresh = cv2.threshold(gray, 150 ,255, cv2.THRESH_BINARY)
showImage(thresh, "threshold")



contours, hierarchy = cv2.findContours(image=thresh,
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_NONE)
                                    
print(contours)

image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours,
                contourIdx=-1, color=(0,255,0), thickness=2,
                lineType=cv2.LINE_AA)


showImage(image_copy, "result")

###################################33


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

    showImage(gradient, "gradient")



#binary
(thresh, im_bw) = cv2.threshold(gradient, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#cv2.imshow("binary", im_bw)
showImage(im_bw, "auto_thresh")


thresh = 110
im_bw_2 = cv2.threshold(gradient, thresh, 255, cv2.THRESH_BINARY)[1]
showImage(im_bw_2, "manual_thresh")




contours, hierarchy = cv2.findContours(image=im_bw_2,
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_NONE)
print()                            
print(contours)

image_copy_2 = image.copy()
cv2.drawContours(image=image_copy_2, contours=contours,
                contourIdx=-1, color=(0,255,0), thickness=2,
                lineType=cv2.LINE_AA)


showImage(image_copy_2, "result2")





