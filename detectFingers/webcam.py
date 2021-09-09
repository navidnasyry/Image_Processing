import cv2 as cv
import numpy as np
import time

'''
ref :
https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

'''

def ShowAndSaveImage(img, pic_name):

    cv.imshow("display image", img)
    k = cv.waitKey()
    if k==ord("s"):
        cv.imwrite(pic_name, img)




def OpenVideo(addr):
    vid_capture = cv.VideoCapture(addr)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    
    else:
        return vid_capture





def main():
    
    vid_capture = cv.VideoCapture("video/Airport.mp4")
    #vid_capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    #get inputs = video metadatas
    # https://docs.opencv.org/4.5.2/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    frame_size = (frame_width, frame_height)
    fps = 10



    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    
    else:
        fps = vid_capture.get(5)
        print("Frames per seco : " , fps , "FPS")


        frame_count = vid_capture.get(7)
        print('Frame count : ' , frame_count)

            
        output = cv.VideoWriter('video/video_output.avi', 
                            cv.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)


    while (vid_capture.isOpened()):

        ret, frame = vid_capture.read()

        if ret == True:
            cv.imshow('frame', frame)
            output.write(frame)

            key = cv.waitKey(50)
            if key == ord('s'):
                break
        else:
            print('Stream disconnected :(')
            break

   
    vid_capture.release()
    cv.destroyAllWindows()


def BGR2GRAY(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def BGR2HSV(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return gray

    

def SkinMask (img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #Change BGR (blue, green, red) image to HSV (hue, saturation, value).

    lower = np.array([0, 48, 80], dtype="uint8")
    #lower range of skin color in HSV.

    upper = np.array([20, 255, 255], dtype= "uint8")
    #upper range of skin color in HSV.

    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    # Detect skin on the range of lower and upper pixel values in the HSV colorspace.

    blurred = cv.blur(skinRegionHSV, (2,2))
    #bluring image to improve masking.

    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    #applying threshing

    return thresh



def FindContours(img):
    thresh = SkinMask(img)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key = lambda x: cv.contourArea(x))

    cv.drawContours(img, [contours], -1, (255,255,0), 2)

    return img
    

def ConvexHull(img):
    thresh = SkinMask(img)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key = lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)

    cv.drawContours(img, [contours], -1, (255,255,0), 2)
    cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    return img



def CountFin(img):
    thresh = SkinMask(img)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key = lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)

    cv.drawContours(img, [contours], -1, (255,255,0), 2)
    cv.drawContours(img, [hull], -1, (0, 255, 255), 2)

    hull = cv.convexHull(contours, returnPoints= False)
    defects = cv.convexityDefects(contours, hull)

    if defects is not None:
      cnt = 0
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
            cv.circle(img, far, 4, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt+1
    cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

    return img




def main2():

    cap = cv.VideoCapture(0)

    while(1):
        
        ret, frame = cap.read()
        frame = SkinMask(frame)

        print(frame)

        cv.imshow('Camera', frame)

        if cv.waitKey(1) == ord('q'):
            break
    

    cap.release()

    cv.destroyAllWindows()



if __name__ == '__main__':
    #main()
    main2()
    


