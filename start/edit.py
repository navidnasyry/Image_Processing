import cv2 as cv
import sys
import numpy as np




def ShowAndSave(img, pic_name):

    cv.imshow("display image", img)
    k = cv.waitKey()
    if k==ord("s"):
        cv.imwrite(pic_name, img)


def cvtColor(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ShowAndSave(gray, "photos/aa3.png")



def fillterColors(img):
    blue, green, red = cv.split(img)

    ShowAndSave(blue, "photos/aa4.png")
    ShowAndSave(green, "photos/aa5.png")
    ShowAndSave(red, "photos/aa6.png")



def fillterColors_2(img):
    blue_channel = np.zeros(img.shape, img.dtype)
    green_channel = np.zeros(img.shape, img.dtype)
    red_channel = np.zeros(img.shape, img.dtype)


    blue, green, red = cv.split(img)
    cv.mixChannels([blue, green, red], [blue_channel], [0,0])

    cv.mixChannels([blue, green, red], [green_channel], [1,1])

    cv.mixChannels([blue, green, red], [red_channel], [2,2])


    ShowAndSave(blue_channel, "photos/aa7.png")
    ShowAndSave(green_channel, "photos/aa8.png")
    ShowAndSave(red_channel, "photos/aa9.png")



def ContrastAndBrightness(img):
    print(img.shape)
    print(img.dtype)
    dest = np.zeros(img.shape, img.dtype)

    alpha = 2.5  # Contrast Control [1.0 - 3.0]
    beta = 3 # Brightness Control [0-100]

    dest = cv.convertScaleAbs(img, alpha, beta)

    ShowAndSave(dest,"photos/aa10.png")


def Laplacian(img):
    lap = cv.Laplacian(img, cv.CV_8UC1)

    lap2 = cv.Laplacian(img, cv.CV_8UC2)
    lap3 = cv.Laplacian(img, cv.CV_8UC3)
    lap4 = cv.Laplacian(img, cv.CV_8UC4)




    ShowAndSave(lap, "photos/aa10.png")
    ShowAndSave(lap2, "photos/aa11.png")
    ShowAndSave(lap3, "photos/aa12.png")
    ShowAndSave(lap4, "photos/aa13.png")

def Blur(img):
    img_blur = cv.GaussianBlur(img, (3,3),sigmaX=0, sigmaY=0)
    ShowAndSave(img_blur, "photos/aa14.png")

def Sobel(img):

    img_blur = cv.GaussianBlur(img, (3,3),sigmaX=0, sigmaY=0)
    sobelx = cv.Sobel(src= img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize= 5)# Sobel Edge Detection on the X axis
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    ShowAndSave(sobelx, "photos/aa14.png")
    ShowAndSave(sobely, "photos/aa15.png")
    ShowAndSave(sobelxy, "photos/aa16.png")
 




def main():
    img = cv.imread('photos/aa.png', 1)
    img2 = cv.imread('photos/aa.png', cv.IMREAD_GRAYSCALE)
    img3 = cv.imread('photos/aa.png', cv.IMREAD_UNCHANGED)

    print(img)
    print(img2)
    print(img3)
    ShowAndSave(img, "photos/aa17.png")
    ShowAndSave(img3, "photos/aa18.png")
    ShowAndSave(img3, "photos/aa19.png")
 

    if img is None:
        sys.exit("Could not read the image.")
    
    cv.imshow("display image", img)
    k = cv.waitKey()

    cvtColor(img)
    fillterColors(img)
    fillterColors_2(img)
    ContrastAndBrightness(img)
    Laplacian(img)
    Blur(img)
    Sobel(img)



    if k==ord("s"):
        cv.imwrite("photos/aa2.png", img)






if __name__ == '__main__':
    main()
    





