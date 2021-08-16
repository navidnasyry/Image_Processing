
import cv2
import matplotlib.pylab as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import argparse



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help= "path to input image")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])





bbox, label, conf = cv.detect_common_objects(image)

print(bbox)
print(label)
print(conf)

output_image = draw_bbox(image, bbox, label, conf)

plt.imshow(output_image)
plt.show()




