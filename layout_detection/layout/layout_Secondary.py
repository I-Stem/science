import os
import cv2
import json
import attr
import requests
import numpy as np
from .utils import image_formating
from .error_handler import LayoutExtractionError
from typing import Dict
import time
import enchant
from .thresholds import text_inside_image
from copy import deepcopy
import layoutparser as lp

@attr.s
class LayoutInference_Secondary:
    """
    Detections for an image using the secondary model, classes ['figure', 'table', 'text']
    Parameters
    ----------
    img_file : bytestream
        image for which you want to predict the layout.
    """

    image_path: str = attr.ib()
    ocr_text: dict = attr.ib()

    @property
    def predict(self,) -> Dict:
        print("USING the secondary layout detector")
        image_array = cv2.imdecode(np.frombuffer(self.image_path, np.uint8), -1)
        model = lp.Detectron2LayoutModel(
        config_path = "./layout/config_Secondary.yaml" , # In model catalog
        label_map = {1:"text", 2:"figure", 3:"table", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}, 
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7] ,# Optional
        )
        page_height, page_width, page_channel = image_array.shape
        predictions=model.detect(image_array)
        bbox = []

        for prediction_ele in predictions:
            if prediction_ele.score  < 0.7:
                continue
            box = list( prediction_ele.coordinates )
            _ = [box[0]/page_width, box[1]/page_height, box[2]/page_width, box[3]/page_height]
            label = prediction_ele.type
            score = prediction_ele.score            
            bbox.append(dict(label=label,bbox=box, relative_box=_, score = score))

        bounding_box = dict(page_width=page_width, page_height=page_height,boxes=bbox)
        original_layout_predictions = deepcopy(bounding_box)
        try:
            layout_bbox = bounding_box
            page_h = self.ocr_text['height']
            page_w = self.ocr_text['width']
            word_dic = []
            word_dic_US = enchant.Dict("en_US")
            word_dic_UK = enchant.Dict("en_UK")
            for lines in self.ocr_text['lines'] :
                for words in lines['words'] : 
                    temp = words['boundingBox']
                    bbox = [temp[0]/page_w,temp[1]/page_h,temp[4]/page_w,temp[5]/page_h]
                    word_dic.append( [ bbox, words['text'] ] )

            threshold = text_inside_image
            for idx, ele in enumerate(layout_bbox['boxes'] ):
                if ele['label'] == 'figure' :
                    text_inside = 0
                    img_box = ele['relative_box']
                    for word_ele in word_dic :
                        word_box = word_ele[0]
                        if word_box[0] >= img_box[0] and word_box[1] >= img_box[1] and word_box[2] <= img_box[2] and word_box[3] <= img_box[3] :# checking each word if it lies inside "image" prediction
                            if word_dic_US.check( word_ele[1] ) or  word_dic_UK.check( word_ele[1] ) : # if after a meaningful word is formed
                                text_inside += 1
                    area = ( img_box[3] - img_box[1] ) * ( img_box[2] - img_box[0] )
                    if text_inside > threshold :
                        if area > .25 :
                            ele['label'] == 'fallback'
                        else :
                            del layout_bbox['boxes'][idx]   
            return layout_bbox          
        except Exception as ex:
            print("Layout inference: Error in layout-detection_Secondary post processing", ex)
            return original_layout_predictions
