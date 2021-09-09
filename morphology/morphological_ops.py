import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")


args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("original" , image)

for i in range(0, 8):
    #erosion
    #eat forground pixels
    eroded = cv2.erode(gray.copy(), None, iterations=i+1)
    cv2.imshow("Eroded {} times".format(i+1), eroded)
    cv2.waitKey(0)
    

cv2.destroyAllWindows()
cv2.imshow("original", image)


for i in range(0, 3):
    #dilation
    #grows our forground region
    dilated = cv2.dilate(gray.copy(), None, iterations=i+1)
    cv2.imshow("Dilated {} times".format(i+1), dilated)
    cv2.waitKey(0)



cv2.destroyAllWindows()
cv2.imshow("original", image)
kernelSize = [(3,3),(5,5),(7,7)]

for kernel_size in kernelSize:
    #opening
    #first an erosion is applied to remove the small blobs,
    #  then a dilation is applied to regrow the size of the original object.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("openning: ({} , {})".format(
        kernel_size[0], kernel_size[1]), opening
    )    

    cv2.waitKey(0)



cv2.destroyAllWindows()
cv2.imshow("original", image)


for kernel_size in kernelSize:
    #closing
    #The exact opposite to an opening would be a closing.
    # A closing is a dilation followed by an erosion.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("openning: ({} , {})".format(
        kernel_size[0], kernel_size[1]), closing
    )    

    cv2.waitKey(0)



cv2.destroyAllWindows()
cv2.imshow("original", image)


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




