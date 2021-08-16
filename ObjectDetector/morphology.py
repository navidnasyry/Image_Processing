

import cv2
import  numpy as np


cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((3,3), np.uint8)

    #erosion and dilation
    erosion = cv2.erode(mask, kernel, iterations= 1)
    dilation = cv2.dilate(mask, kernel, iterations= 1)

    #openning and closing
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    ###################################


    contours, hierarchy = cv2.findContours(image=mask,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)
 

    frame_copy = frame.copy()
    cv2.drawContours(image=frame_copy, contours=contours,
                    contourIdx=-1, color=(0,255,0), thickness=2,
                    lineType=cv2.LINE_AA)


    ###################################


    contours, hierarchy = cv2.findContours(image=erosion,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)
    print()                            
    print(contours)

    erosion_copy = frame.copy()
    cv2.drawContours(image=erosion_copy, contours=contours,
                    contourIdx=-1, color=(0,255,0), thickness=2,
                    lineType=cv2.LINE_AA)



    ###################################


    contours, hierarchy = cv2.findContours(image=dilation,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)


    dilation_copy = frame.copy()
    cv2.drawContours(image=dilation_copy, contours=contours,
                    contourIdx=-1, color=(0,255,0), thickness=2,
                    lineType=cv2.LINE_AA)


    ###################################


    contours, hierarchy = cv2.findContours(image=opening,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)

    opening_copy = frame.copy()
    cv2.drawContours(image=opening_copy, contours=contours,
                    contourIdx=-1, color=(0,255,0), thickness=2,
                    lineType=cv2.LINE_AA)


    ###################################


    contours, hierarchy = cv2.findContours(image=closing,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)

    closing_copy = frame.copy()
    cv2.drawContours(image=closing_copy, contours=contours,
                    contourIdx=-1, color=(0,255,0), thickness=2,
                    lineType=cv2.LINE_AA)




    ##################################



    


    cv2.imshow('Original',frame)

    cv2.imshow("Mask_2", frame_copy)
    cv2.imshow('Mask',mask)

    cv2.imshow('Erosion',erosion)
    cv2.imshow("Erosion_2", erosion_copy)


    cv2.imshow('Dilation',dilation)
    cv2.imshow('Dilation_2', dilation_copy)


    cv2.imshow('opening', opening)
    cv2.imshow('opening_2', opening_copy)

    cv2.imshow('closing', closing)
    cv2.imshow('closing_2', closing_copy)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()

