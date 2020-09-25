import numpy as np
from PIL import Image
import math


def iou(box, boxes, isMin = False):
    box_area = (box[2]-box[0])*(box[3]-box[1]) #[x1,y1,x2,y2,c]
    boxes_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]- boxes[:,1])
    xx1 = np.maximum(box[0],boxes[:,0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0,(xx2-xx1))
    h = np.maximum(0,(yy2-yy1))

    inter = w*h
    if isMin:
        #smallest area
        over = np.true_divide(inter,np.minimum(box_area,boxes_area))
    else:
        #union
        over = np.true_divide(inter,(boxes_area+ box_area-inter))
    return over


def nms(boxes, threshold = 0.3, isMin = False):
    if boxes.shape[0] == 0:
        return np.array([])
    #arrange with c
    _boxes = boxes[(-boxes[:,4]).argsort()]
    r_boxes = []
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        index = np.where(iou(a_box,b_boxes,isMin) < threshold)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)


def convert_to_square(bbox):
    #input must be float
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:,3]- bbox[:,1]
    w = bbox[:,2]- bbox[:,0]
    max_size = np.maximum(w,h)
    square_bbox[:, 0] = square_bbox[:, 0] - (max_size - w) / 2
    square_bbox[:, 1] = square_bbox[:, 1] - (max_size - h) / 2
    square_bbox[:, 2] = square_bbox[:, 0] + max_size
    square_bbox[:, 3] = square_bbox[:, 1] + max_size
    return square_bbox



def convert_to_square_shrink(bbox):
    #input must be float
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:,3]- bbox[:,1]
    w = bbox[:,2]- bbox[:,0]
    max_size = np.minimum(w,h)
    square_bbox[:, 0] = square_bbox[:, 0] - (max_size - w) / 2
    square_bbox[:, 1] = square_bbox[:, 1] - (max_size - h) / 2
    square_bbox[:, 2] = square_bbox[:, 0] + max_size
    square_bbox[:, 3] = square_bbox[:, 1] + max_size
    return square_bbox

def allign_face_angle(bbox):
    angle_e = math.atan((bbox[8]-bbox[6])/(bbox[7]-bbox[5]))
    # print(angle_e * 180 / math.pi)
    angle_m = math.atan((bbox[14]-bbox[12])/(bbox[13]-bbox[11]))
    # print(angle_m * 180 / math.pi)
    return ((angle_e+angle_m)/2)

def rotate_bbox(cx,cy,x1,y1,x2,y2,rotate_angle):
    d1 = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    d2 = math.sqrt((cx - x2) ** 2 + (cy - y2) ** 2)
    if y1 < cy:
        x1 = x1 - d1 * math.sin(rotate_angle)
        y1 = y1 + d1 * math.sin(rotate_angle)
    else:
        x1 = x1 + d1 * math.sin(rotate_angle)
        y1 = y1 - d1 * math.sin(rotate_angle)
    if y2 < cy:
        x2 = x2 - d1 * math.sin(rotate_angle)
        y2 = y2 + d1 * math.sin(rotate_angle)
    else:
        x2 = x2 + d2 * math.sin(rotate_angle)
        y2 = y2 - d2 * math.sin(rotate_angle)
    return(x1,y1,x2,y2)






def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def draw_circle(x,y,pixel = 5):
    x1 = x-pixel
    y1 = y-pixel
    x2 = x+pixel
    y2 = y+pixel
    return(x1,y1,x2,y2)


