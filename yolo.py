import os.path
import cv2
import numpy as np
import requests

# download yolo net config file
yolo_config  = 'yolov3.cfg'
if not os.path.isfile(yolo_config):
    url =  'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
    r = requests.get(url)
    with open (yolo_config, 'wb') as f:
        f.write(r.content)

# download yolo net weights
yolo_weights = 'yolov3.weights'
if not os.path.isfile(yolo_weights):
    url =  'https://pjreddie.com/media/files/yolov3.weights'
    r = requests.get (url)
    with open (yolo_weights, 'wb') as f:
        f.write(r.content)

# download class names file 
classes_file = 'coco.names'
if not os.path.isfile(classes_file):
    url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    r = requests.get(url)
    with open (classes_file, 'wb') as f:
        f.write(r.content)

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


image_file = 'source.jpg'
if not os.path.isfile(image_file):
    url = "https://upload.wikimedia.org/wikipedia/commons/1/17/Haifa_Crosswalk_by_David_Shankbone.jpg"
    r = requests.get(url)
    with open(image_file, 'wb') as f:
        f.write(r.content)
    
image = cv2.imread(image_file)
blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0),
True, crop=False)


net = cv2.dnn.readNet(yolo_weights, yolo_config)


net.setInput(blob)


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in 
net.getUnconnectedOutLayers() ]

outs = net.forward(output_layers)


class_ids = list ()
confidences = list()
boxes = list()


for out in outs:
    for detection in out:
        center_x = int(detection[0] * image.shape[1])
        center_y = int(detection[1] * image.shape[0])
        w = int(detection[2] * image.shape[1])
        h = int(detection[3] * image.shape[0])
        x = center_x - w // 2
        y = center_y - h // 2
        boxes.append([x, y, w, h])

        
        class_id = np.argmax(detection[5:])
        class_ids.append(class_id)

        # confidence 
        confidence = detection[4]
        confidences.append(float (confidence))


ids = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3,
nms_threshold=0.5)


colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in ids:
    i = i[0]
    x, y, w, h = boxes[i]
    class_id = class_ids[i]

    color = colors[class_id]

    cv2.rectangle(image, (round(x), round(y)), (round(x + w),
round(y + h)), color, 2)

    label = "%s: %.2f" % (classes[class_id], confidences[i])
    cv2.putText(image, label, (x - 10, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow("Object detection", image)
cv2.waitKey()