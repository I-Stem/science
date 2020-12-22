"""
Basic utility for the all the files
"""
import io
import cv2
import time
import json
import numpy as np
from PIL import Image
from copy import deepcopy
from shapely.geometry import Polygon

def iou(b1, b2): # bounding boxes in [x1,y1,x2,y2] format
    assert b1[0] < b1[2]
    assert b1[1] < b1[3]
    assert b2[0] < b2[2]
    assert b2[1] < b2[3]
    # determine the coordinates of the intersection rectangle
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1 [2], b2[2])
    y_bottom = min(b1[3], b2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0 # no instersection   
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    bb2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def json_reader(file):
    """
    json file roader
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


def get_relative_position(wid, hei, bbox):
    """
    convert the boundig box from relative (x1, y1, x2, y2 ) to exact (x, y, w, h) corresponding to width and height
    """
    _ = deepcopy(bbox)
    bx = _["relative_box"]
    bx[0] = int(bx[0] * wid)
    bx[1] = int(bx[1] * hei)
    bx[2] = int(bx[2] * wid) - bx[0]
    bx[3] = int(bx[3] * hei) - bx[1]
    bbox["bbox_new"] = bx
    return bbox


def get_line_box(bbox):
    """
    line box for the words given
    ** maybe max for more values
    """
    return_list = []
    bbox = [i["boundingBox"] for i in bbox]
    array = np.array(bbox)
    for i in range(0, 8):
        arr = array[0:, i]
        value = min(arr)
        if i in [2, 4,5,7]: # [2,4,5,7]
            value = max(arr)
        return_list.append(value)
    return return_list


def overlap(rect1, rect2): # wrong
    """
    If two bounding box are overlapping or not.
    """
    try:
        p1 = Polygon(
            [
                (rect1[0], rect1[1]),
                (rect1[2], rect1[1]),
                (rect1[2], rect1[3]),
                (rect1[0], rect1[3]),
            ]
        )
        p2 = Polygon(
            [
                (rect2[0], rect2[1]),
                (rect2[2], rect2[1]),
                (rect2[2], rect2[3]),
                (rect2[0], rect2[3]),
            ]
        )
        return p1.intersects(p2)
    except:
        return True


def repeating_fixer(l1):
    image_data = []
    table_data = []
    for key in sorted(list(l1.keys())):
        i = l1[key]
        img = i["image_bbox"]
        tb = i["table_bbox"]
        d = {}
        for k, v in img.items():
            if v not in image_data:
                d[k] = v
                image_data.append(v)
            else:
                if k in i["text"]:
                    i["text"].remove(k)

        d1 = {}
        for k, v in tb.items():
            if v["box"] not in table_data:
                d1[k] = v
                table_data.append(v["box"])
            else:
                if k in i["text"]:
                    i["text"].remove(k)

        i["image_bbox"] = d
        i["table_bbox"] = d1
        l1[key] = i
    return l1


def remove_duplication(table_data, image_data):
    """
    Helpful for the if multiple boxes are overlapping with each other,
    removes img bboxs which overlap with table bbox
    """
    new_image_data = []
    for img in image_data:
        add = True
        x1, y1, w, h = img["bbox_new"]
        x2, y2 = x1 + w, y1 + h
        bbox1 = [x1, y1, x2, y2]
        for tab in table_data:
            x1, y1, w, h = tab["bbox_new"]
            x2, y2 = x1 + w, y1 + h
            bbox2 = [x1, y1, x2, y2]
            if overlap(bbox1, bbox2):
                add = False
                break
        if add:
            new_image_data.append(img)
    return new_image_data


def image_formating(img):
    """
    It reduce the image size
    """
    shape = img.shape
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        scale_percentage = int((2000 / img.shape[0]) * 100)
        width = int(img.shape[1] * scale_percentage / 100)
        height = int(img.shape[0] * scale_percentage / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def image_to_byte_array(image: Image):
    """
    convert the byte stream image into array.
    """
    output = io.BytesIO()
    image.save(output, format="JPEG")
    data = output.getvalue()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    img = image_formating(img)
    img = cv2.imencode(".jpg", img)[1].tostring()
    return img
