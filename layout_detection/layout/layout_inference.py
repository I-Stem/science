import os
import cv2
import json
import attr
import requests
import numpy as np
from ast import literal_eval
from .utils import image_formating
from .error_handler import LayoutExtractionError
from typing import Dict
import time


@attr.s
class LayoutInference:
    """
    Layout inference detections for an image
    Parameters
    ----------
    image_path : bytestream
        image for which you want to predict the layout.
    """

    image_path: str = attr.ib()
    ocr_text: dict = attr.ib()
    url: None = attr.ib(repr=False, init=False)

    def __attrs_post_init__(self):
        self.url = os.getenv("LAYOUT_API_URL")

    @property
    def predict(self,) -> Dict:
        """
        It predict the layout for the image and return dictionary
        
        return output is : 
        {
        'page_width': 1530,
        'page_height': 1980,
        'boxes':
            [
                {'label': 'text',
                'bbox': [768, 670, 1352, 836],
                'relative_box': [0.5019607843137255,
                    0.3383838383838384,
                    0.8836601307189542,
                    0.4222222222222222]
                },
                {'label': 'table',
                'bbox': [769, 160, 1311, 651],
                'relative_box': [0.5026143790849673,
                    0.08080808080808081,
                    0.8568627450980392,
                    0.3287878787878788]
                }
            ]
        }
        """
        img = cv2.imdecode(np.frombuffer(self.image_path, np.uint8), -1)
        img = image_formating(img) # converts image to around 10 MB
        h, w, c = img.shape
        try:
            headers = {"Content-Type": "application/json"}
            input_json = json.dumps({ "data": img.tolist() }) # input_json is approx 50 MB in size
            for i in range(1,4): # number of tries = 3
                print("Layout inference: Calling layout API try ",str(i) )
                
                try:
                    predictions = requests.post( self.url, input_json, headers=headers, timeout=100 )
                    bbox = json.loads( predictions.content )
                    if "boxes" in bbox :
                        print ("Layout inference: Successful ")
                        break
                    time.sleep( 5*i )
                except Exception as ex:
                    print ("Layout inference: Error in layout API call ",ex)

            del img
            return literal_eval(bbox)

        except Exception as ex:
            print("Layout inference: layout extraction using layout API is failing", ex)
            raise LayoutExtractionError
