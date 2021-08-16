
import cv2 as cv
import numpy as np



def ShowAndSave(img, pic_name):
    
    cv.imshow("display image", img)
    k = cv.waitKey()
    if k==ord("s"):
        cv.imwrite(pic_name, img)



protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

print(net)
print(type(net))

frame = cv.imread("photos/aa.png")

inWidth = 634
inHeight = 951

inpBlob = cv.dnn.blobFromImage(frame, 
                                1.0 / 255,
                                (inWidth, inHeight),
                                (0 , 0 , 0),
                                swapRB= False,
                                crop= False)

print(inpBlob)

net.setInput(inpBlob)

print(net)

output = net.forward()

print(type(output))
print(output)


H = output.shape[2]
W = output.shape[3]

points = []

for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv.minMaxLoc(probMap)


    x = (inWidth * point[0]) / W
    y = (inHeight * point[1]) / H

    threshold = 0.1
    if prob > threshold :
        cv.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
        cv.putText(frame, "{}".format(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)


frameCopy = frame
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv.line(frameCopy, points[partA], 
                points[partB], (0, 255, 0), 3)


ShowAndSave(frame, "photos/out1.png")

ShowAndSave(frameCopy, "photos/out2.png")





