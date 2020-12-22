"""
Using OCR output and Bounding Box, return image into json readable format.

"""
import os
import sys
import cv2
import time
import json
import attr
import numpy as np
from typing import Dict
from .utils import overlap
from .utils import get_relative_position
from .error_handler import ImageRecreationError
from .utils import get_line_box, image_formating
from .table_extraction import get_table_csv_results
from .utils import repeating_fixer, remove_duplication, iou
from copy import deepcopy


@attr.s
class ImageRecreation:
    """
    Using the ocr output and bounding box, it recreate the document into readable function
    Input :
    ocr_text: dict
        ocr output

    bounding_box: dict
        bbox predicted by layout prediction model

    img_file: str
        image file for which we want to recreate

    page_type: str (Optional)
        Image is single column or multi column layout 
    """

    ocr_text: Dict = attr.ib()
    bounding_box: Dict = attr.ib()
    img_file: str = attr.ib()
    page_type: str = attr.ib(default=None)
    title_area: list = attr.ib(repr=False, init=False)
    text_area: list = attr.ib(repr=False, init=False)

    def __attrs_post_init__(self):
        self.title_area = []
        self.text_area = []
        self.junk = [] # unnecessary

    @property
    def extract(self):
        boxes = self.bounding_box["boxes"]
        temp = self.ocr_text # temp -> ocr_text
        full_text = [i["text"] for i in temp["lines"]] # full_text is the last ele of json ( contains all the ocr text lines )
        try:
            wid, hei = temp["width"], temp["height"]
            updated_bbox = [get_relative_position(wid, hei, b) for b in boxes]
            sorted_data = sorted(updated_bbox, key=lambda i: i["bbox"][1])
            img_data = [i for i in sorted_data if i["label"] in ["figure"]]
            table_data = [i for i in sorted_data if i["label"] in ["table"]]
            self.junk = table_data
            self.title_area = [i for i in sorted_data if i["label"] in ["title"]]
            self.text_area = [i for i in sorted_data if i["label"] in ["text", "list"]]
            if len(table_data):
                print("Recreation: table extraction")
                table_data = get_table_csv_results(self.img_file)
                table_data = [get_relative_position(wid, hei, b) for b in table_data]
                self.junk += img_data
                img_data = remove_duplication(table_data, img_data)

            self.page_type = self._get_page_type(updated_bbox, wid)
            if self.page_type == "two-column":
                print("Recreation: Processing page with two columns")
                text = self.get_two_columns_text(temp, img_data, table_data, wid, hei)
            else:
                text = self.get_one_columns_text(
                    temp, img_data, table_data, self.text_area
                )

        except Exception as ex:
            raise ImageRecreationError
        text["ocr_result"] = full_text
        return text

    def _get_page_type(self, bbox: list, wid: int) -> str:
        """
        It return the page type 'single' or 'two-column'.
        If difference between max and min text area's starting x coordinate > page_width/ 2 then two-column
        """
        if self.page_type != None:
            return self.page_type
        page_type = "two-column"
        x_min, x_max = self._get_index_value(bbox, wid)
        if x_max == 100000: # only one text area found
            page_type = "single"
        elif x_max - x_min < (int(wid / 2) - x_min):
            page_type = "single"
        return page_type

    def _get_index_value(self, bbox: list, thr: int, index: int = 0) -> tuple:
        """
        It  is helper function for the _get_page_type.
        returns min and max value of the starting x coord of text areas
        """
        x_value = [
            i["bbox_new"][index]
            for i in bbox
            if (
                i["label"] not in ["title", "figure", "table"] # using "list" and "text" elements
                and i["bbox_new"][-1] > 20 #?
            )
        ]
        if len(x_value) < 2:
            x_value = [0, 100000]
        return min(x_value), max(x_value)

    def get_two_columns_text(self, temp, img_data, table_data, wid, hei):
        """
        It return the recreationed json for the two_columns page/image. 
        """
        # temp - complete ocr's output 
        col_1, col_2 = self.column_wise_data(temp, img_data, table_data, wid, hei)
        text_1 = self.get_one_columns_text(
            col_1, col_1["image_data"], col_1["table_data"], col_1["bbox"]
        )
        text_2 = self.get_one_columns_text(
            col_2, col_2["image_data"], col_2["table_data"], col_2["bbox"]
        )
        list_data = dict(col_1=text_1, col_2=text_2)
        return repeating_fixer(list_data)

    def find_figure_index(
        self, image_data, image_index, line_box, image_added
    ):
        """
        Helpful for finding the image position into the image.
        if an image overlaps with an ocr detected line or lies above a line then it is added.
        ocr detected line -> can be gibberish detected by ocr in image or ocr prediction on a table
        """
        new_x = line_box[0] # ?
        new_y = line_box[1]
        #old_y = prev_line_box[5]
        text = None
        bbox = None
        if len(image_data) == 0:
            return (text, bbox, image_index)
        for img in image_data:
            im_x1 = img["bbox_new"][0]
            im_x2 = im_x1 + img["bbox_new"][2]
            im_y1 = img["bbox_new"][1]
            im_y2 = im_y1 + img["bbox_new"][3]
            if (
                (new_y >= im_y1 and new_y <= im_y2) or (new_y > im_y2 and new_y > im_y1) # anything greater than y2 then will be greater than y1
            ) and not img["bbox"] in image_added:
                text = f"image_{str(image_index)}.jpg"
                bbox = img["bbox"]
                image_index += 1
                break

        return text, bbox, image_index

    def find_table_index(
        self, table_data, table_index, line_box, table_added
    ):

        """
        Helpful for finding the table position into the image.
        """
        (
            X_top_left,
            Y_top_left,
            X_top_right,
            Y_top_right,
            X_bottom_right,
            Y_bottom_right,
            X_bottom_left,
            Y_bottom_left,
        ) = line_box
        bbox1 = [X_top_left, Y_top_left, X_bottom_right, Y_bottom_right]
        text = None
        bbox = None
        table_text = None
        if len(table_data) == 0:
            return (text, bbox, table_index, table_text)
        for img in table_data:
            im_x1 = img["bbox_new"][0] #?
            im_x2 = im_x1 + img["bbox_new"][2] #?
            im_y = img["bbox_new"][1] #?
            im_y1 = img["bbox_new"][1] #?
            im_y2 = im_y1 + img["bbox_new"][3] #?
            x1, y1, w, h = img["bbox_new"]
            x2, y2 = x1 + w, y1 + h
            if overlap(bbox1, [x1, y1, x2, y2]) and img["bbox_new"] not in table_added:
                text = f"table_{str(table_index)}.jpg"
                table_text = img["text"]
                bbox = img["bbox_new"]
                table_index += 1
                break
        return text, bbox, table_index, table_text

    def find_paragraph_index(self, line, text_area):
        """
        Helpful for finding the paragraph index into the image.
        """
        is_para = False
        para_bbox = None
        line_box = line["boundingBox"]
        (
            X_top_left,
            Y_top_left,
            X_top_right,
            Y_top_right,
            X_bottom_right,
            Y_bottom_right,
            X_bottom_left,
            Y_bottom_left,
        ) = line_box
        bbox1 = [X_top_left, Y_top_left, X_bottom_right, Y_bottom_right]
        for index, i in enumerate(text_area):
            x1, y1, w, h = i["bbox_new"]
            x2, y2 = x1 + w, y1 + h
            if overlap(bbox1, [x1, y1, x2, y2]):
                para_bbox = [x1, y1, x2, y2]
                is_para = True
                break
        return is_para, para_bbox

    def find_title(self, line):
        """
        Helpful for finding the text position into the image.
        """
        line_box = line["boundingBox"]
        (
            X_top_left,
            Y_top_left,
            X_top_right,
            Y_top_right,
            X_bottom_right,
            Y_bottom_right,
            X_bottom_left,
            Y_bottom_left,
        ) = line_box
        bbox1 = [X_top_left, Y_top_left, X_bottom_right, Y_bottom_right]
        is_title = False
        bbox = None
        for i in self.title_area:
            x1, y1, w, h = i["bbox_new"]
            x2, y2 = x1 + w, y1 + h
            if overlap([x1, y1, x2, y2], bbox1):
                is_title = True
                bbox = [x1, y1, x2, y2]
                break
        return is_title, bbox

    def get_one_columns_text(
        self,
        temp,
        img_data,
        table_data,
        para_data,
        image_index=0,
        table_index=0,
        para_index=0, #?
    ):
        """
        It return the recreationed json for the one columns page/image. 
        """
        doc = []
        para_index = 0
        para_bbox = None
        image_added = []
        table_added = []
        image_bbox = {}
        table_bbox = {}
        #prev_line_box = [0] * 8
        for line in temp["lines"]:
            line_box = line["boundingBox"]
            img_text, img_bbox, image_index = self.find_figure_index(
                img_data, image_index, line_box, image_added
            )
            tab_text, tab_bbox, table_index, t_text = self.find_table_index(
                table_data, table_index, line_box, table_added
            )
            if img_text != None:
                print("Recreation: Image within document being processed")
                image_bbox[img_text] = img_bbox
                doc.append(img_text)

            if tab_text != None:
                print("Recreation: Table being processed")
                table_bbox[tab_text] = dict(box=tab_bbox, text=t_text)
                doc.append(tab_text)

            if img_bbox != None:
                image_added.append(img_bbox)

            if tab_bbox != None:
                table_added.append(tab_bbox)

            valid = self.avoid_overlapping(line_box, img_data + table_data + self.junk)
            if valid:
                is_title, tbox = self.find_title(line)
                if (
                    is_title and tbox != None
                    #and para_bbox != None # this excludes marking first line as title
                   # and abs(para_bbox[0] - tbox[0]) < 50
                ):
                    print("Recreation: Title detected and being processed")
                    doc.append((line["text"], -1))
                else:
                    is_title = False

                if not is_title:
                    is_para, temp = self.find_paragraph_index(line, para_data)
                    if is_para: # new para created if line lies in new text area ( list + text ) found else lines appended to prev para
                        if temp != para_bbox:
                            para_index += 1
                            para_bbox = temp
                        doc.append((line["text"], para_index))
                    else:
                        doc.append((line["text"], para_index))

            #prev_line_box = line_box
        return_dict = dict(text=doc, image_bbox=image_bbox, table_bbox=table_bbox)
        if self.page_type == "single":
            return_dict = dict(col_1=return_dict)
        return return_dict

    def is_overlapping(self, data):
        """
        It find the overlapping of the bounding box.
        """
        return_value = False
        for index_1, i in enumerate(data[:-1]):
            is_overap = False
            for index_2, j in enumerate(data[index_1 + 1 :]):
                if overlap(i["bbox"], j["bbox"]):
                    is_overap = True
                    break
            if is_overap:
                return_value = True
                break
        return return_value

    def _get_line_type(self, line_box, thresold, height):
        """
        if line starts before threshold (atleast 20 pixel before) and ends after threshold then single
        if the line lies purely in one col then it returns double
        if the line lies in two or more para index then it's double
        It file the line is two column or single column. 
        Sometime in the two-columns page, if table is in the center the table caption should belongs to the single column.
        """
        line_type = "double"
        (
            X_top_left,
            Y_top_left,
            X_top_right,
            Y_top_right,
            X_bottom_right,
            Y_bottom_right,
            X_bottom_left,
            Y_bottom_left,
        ) = line_box
        bbox1 = [X_top_left, Y_top_left, X_bottom_right, Y_bottom_right]

        if X_top_right > thresold and (
            X_top_left < thresold and abs(X_top_left - thresold) > 20
        ):
            line_type = "single"
            temp = []
            for i in self.text_area:
                x1, y1, w, h = i["bbox_new"]
                x2, y2 = x1 + w, y1 + h
                if overlap(bbox1, [x1, y1, x2, y2]):
                    temp.append(i)
            if len(temp) > 1 and not self.is_overlapping(temp): # if line overlaps with multiple text areas and none of them overlap with each other
                line_type = "double"
        return line_type

    def avoid_overlapping(self, line_box, img_data): # what if line_box lies on upper side
        """
        if image contains the text, then to avoid that junk, this is useful.
        """
        valid = True
        x1_new, y1_new, x2_new, y2_new = (
            line_box[0],
            line_box[1],
            line_box[2],
            line_box[-1],
        )
        for i in img_data:
            x, y, w, h = i["bbox_new"]
            if (
                y1_new > y
                and y1_new < y + h
                and (x1_new > x and (x2_new < x + w or abs(x2_new - (x + w)) < 30))
            ):
                valid = False
                break
        return valid

    def divide_into_two_columns(self, data, thresold):
        img_box_1 = []
        img_box_2 = []
        for i in data:
            if ( i["bbox_new"][0] < thresold and (i["bbox_new"][0] + i["bbox_new"][2]) > thresold ):
                i_box= deepcopy( i["bbox_new"])
                i_box=[i_box[0],i_box[1],i_box[0]+i_box[2],i_box[1]+i_box[3]]
                thresh_box = deepcopy(i_box)
                thresh_box[2] = thresold
                if iou(i_box,thresh_box) >.25 :
                    img_box_1.append(i)
                img_box_2.append(i) # will be removed from col 2 by repeating_fixer 
            elif i["bbox_new"][0] < thresold:
                img_box_1.append(i)
            else:
                img_box_2.append(i)
        return img_box_1, img_box_2

    def _get_thresold( self, wid, hei):
        """
        X coord of text box, with least distance to mid point of the page
        """
        if len( self.text_area ) == 0:
            return int((wid / 2)*.95)
        try:
            index = np.argmin(
                [abs(i["bbox_new"][0] - (int((wid / 2)*.95))) for i in self.text_area]
            )
            return self.text_area[index]["bbox_new"][0]
        except Exception as ex:
            print("Recreation: error detected")
            print(ex)
            return int((wid / 2)*.95)

    def column_wise_data(self, temp, image_data, table_data, wid, hei):
        """
        It convert the two column page information into single column formats.
        """
        box_1 = [] # text for col1 in ocr format (text,words,bounding box)
        box_2 = []
        thresold = self._get_thresold(wid, hei)
        for line in temp["lines"]:
            col_1 = ""
            col_2 = ""
            word_list_1 = []
            word_list_2 = []
            if self._get_line_type(line["boundingBox"], thresold, hei) == "single":
                box_1.append(line)
                continue
            # text for non single lines ( lines that dont extend across both side of threshold ), is divided on there word level bounding boxes
            for word in line["words"]: # if the word starts and ends before threshold it goes in col1
                if not (
                    word["boundingBox"][0] > thresold
                    or word["boundingBox"][2] > thresold
                ):
                    col_1 += " " + word["text"]
                    word_list_1.append(word)
                else:
                    col_2 += " " + word["text"]
                    word_list_2.append(word)

            if len(word_list_1):
                if len(word_list_2) == 0:
                    line_box = line["boundingBox"]
                else:
                    line_box = get_line_box(word_list_1)
                box_1.append(dict(text=col_1, words=word_list_1, boundingBox=line_box))

            if len(word_list_2):
                if len(word_list_1) == 0:
                    line_box = line["boundingBox"]
                else:
                    line_box = get_line_box(word_list_2)
                box_2.append(dict(text=col_2, words=word_list_2, boundingBox=line_box))

        col_1_image, col_2_image = self.divide_into_two_columns(image_data, thresold)
        col_1_table, col_2_table = self.divide_into_two_columns(table_data, thresold)
        col_1_text, col_2_text = self.divide_into_two_columns(self.text_area, thresold)
        col_1 = dict(
            lines=box_1, image_data=col_1_image, table_data=col_1_table, bbox=col_1_text
        )
        col_2 = dict(
            lines=box_2, image_data=col_2_image, table_data=col_2_table, bbox=col_2_text
        )
        return col_1, col_2
