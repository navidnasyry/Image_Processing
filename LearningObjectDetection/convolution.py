

import cv2
import argparse
import numpy as np 




def showImage(img, name):

    #cv2.imshow("original", image)
    winname = name
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    pic_name = "image_"+ name +".png"
    cv2.imwrite(pic_name, img)
    cv2.waitKey(0)



#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, 
 #               help= "path to input image")

#args = vars(ap.parse_args())



img = cv2.imread("cat.jpg")
showImage(img, "original")


#Identity kernel
kernel1 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]]
                    )

im1 = cv2.filter2D(img, -1, kernel1)
showImage(im1 ,'Identity kernel' )



#shapening kernel
kernel2 = np.array([[-1,-1,-1],
                    [-1, 9,-1],
                    [-1,-1,-1]]
                    )
                    
im2 = cv2.filter2D(img, -1, kernel2)
showImage(im2, 'Sharpening kernel')


#edge
kernel2 = np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]]
                    )
                    
                    
im2 = cv2.filter2D(img, -1, kernel2)
showImage(im2, 'edge')


#blurring kernel
kernel3 = np.array([[0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816],
                    [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,0.02040816, 0.02040816]])

                    
im3 = cv2.filter2D(img, -1, kernel3)
showImage(im3, 'Blurring kernel')