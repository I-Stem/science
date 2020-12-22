"""

Take the input given by recreation.py file and convert into meaningful json file.


"""
import os
import io
import cv2
import json
import docx
import base64
import numpy as np
from PIL import Image
from docx.shared import Inches
from docx.enum.text import WD_LINE_SPACING
from .error_handler import FormatingError
from typing import Dict


def crop(image_path, coords):
    """
    Crop the original image and return the crop image based on given coordinates.

    Parameters
    ----------
        image_path : bytestream(image)
            the  original image which you want to crop

        coords : list/tuple
            It is tuple which contains x1,y1,x2,y2
    
    Returns
    -------
        cropped base64 formated image


    """
    # x,y,w,h = coords
    # x1,y1,x2,y2 = x-10,y-10, x+w+10, y+h+10
    x1, y1, x2, y2 = coords
    x1, y1, x2, y2 = x1, y1, x2, y2
    # y1, y2 = y1-30, y2+200
    image_obj = Image.open(io.BytesIO(image_path))
    cropped_image = image_obj.crop([x1, y1, x2, y2])
    buffer = io.BytesIO()
    cropped_image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    data_uri = base64.b64encode(buffer.read()).decode("ascii")
    return data_uri


def column_formating(image_path, data):
    samples = []
    img = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    # img = cv2.imread(image_path)
    for index, i in enumerate(data["text"]):
        tmp = None
        if isinstance(i, str) and i.startswith("table") and i.endswith(".jpg"):
            print("Formatting: getting table ready for final json")
            table_data = data["table_bbox"][i]["text"]
            records = table_data.values.tolist()
            tmp = dict(
                idx=index,
                paragraph_id=-1,
                type="table",
                font_size=None,
                font_family=None,
                font_color=None,
                alignment=None,
                bold=None,
                italic=None,
                underline=None,
                block=[dict(content=records)],
            )

        if isinstance(i, str) and i.startswith("image") and i.endswith(".jpg"):
            print("Formatting: getting image ready for final json")
            bbox = data["image_bbox"][i]
            image_data = crop(image_path, bbox)
            tmp = dict(
                idx=index,
                paragraph_id=-1,
                type="image",
                font_size=None,
                font_family=None,
                font_color=None,
                alignment=None,
                bold=None,
                italic=None,
                underline=None,
                block=[dict(content=image_data)],
            )

        if tmp == None:
            text_type = "p"
            text, para_ind = i
            if para_ind == -1:
                text_type = "title"
            tmp = dict(
                idx=index,
                paragraph_id=para_ind,
                type=text_type,
                font_size=None,
                font_family=None,
                font_color=None,
                alignment=None,
                bold=None,
                italic=None,
                underline=None,
                block=[dict(content=text)],
            )
        samples.append(tmp)
    return samples


def json_formating(image_path, result):
    """
    convert the output of recreation file into more meaningful format.

    Parameters:
    ----------
        image_path : bytestrem
        file for which you want to do OCR

        result : dict
        output of the recreation file
    
    Return:
    -------
        list of dictionary.
        each dictionary object looks like : 
        dict(
                idx=index,
                paragraph_id=-1,
                type=["image"/"text"/"table"],
                font_size=None,
                font_family=None,
                font_color=None,
                alignment=None,
                bold=None,
                italic=None,
                underline=None,
                block=[dict(content=image_data)],
            )
    """
    columns = []
    for i, j in result.items():
        if i == "ocr_result":
            columns.append(dict(idx=-1, content=dict(full_text=j)))
            continue
        column_index = i.split("_")[-1]
        idx = int(column_index)
        columns.append(dict(idx=idx, content=column_formating(image_path, j)))
    return columns
