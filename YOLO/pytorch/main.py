''' 
    ref:
    https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python

'''


import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet


nms_threshold = 0.6

iou_threshold = 0.4

cfg_file = "cfg/yolov3.cfg"
weigh_file = "../../../yolov3.weights"
namesfile = "data/coco.names"

m = Darknet(cfg_file)
m.load_weights(weigh_file)
class_names = load_class_names(namesfile)
m.print_network()

original_image = cv2.imread("images/city_scene.jpg")
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

img = cv2.resize(image_rgb, (m.width, m.height))


boxes = detect_objects(m, img, iou_threshold, nms_threshold)


plot_boxes(original_image, boxes, class_names, plot_labels=True)



