"""
It handles complete end-to-end process.
It takes, image/pdf as input and perform following step
    1. DO the ocr of the input file
    2. layout detection
    3. layout recreating -> convert scan document into readable formate.
    4. Apply formating to encode more information about the image(helpful for converting json to html,text, etc formate)
"""
import os
import cv2
import attr
import numpy as np
from dotenv import load_dotenv
from .ocr import image_to_text
from .recreation import ImageRecreation
from .layout_inference import LayoutInference
from .error_handler import LayoutExtractionError
from typing import Dict, List, Any, Union
from .formating import json_formating
from .utils import image_formating, image_to_byte_array
from pdf2image import convert_from_bytes
from .layout_fixer import layout_fixer,  line_break_fixer
from .post_processing import post_processing
from .list_handling import list_handling
from .math_ocr import math_ocr
from .fallback_ocr import fallback_ocr

@attr.s
class LayoutOcr:
    """
    Ocr using the Layout.

    Input
    -----
    image_path : str
        File object  of the input file for which you want to OCR

    env_file_path : str
        .env file path to setup the api   

    input_type : str
        it can be image and pdf
    """
    image_path: str = attr.ib()
    env_file_path: str = attr.ib( default=None )
    input_type: str = attr.ib( default="image" )
    doc_type: str = attr.ib( default="doc_type" )

    def __attrs_post_init__(self):
        if self.env_file_path == None:
            load_dotenv()
        else:
            load_dotenv(self.env_file_path)

    @property
    def result(self) -> Dict:
        """
        INPUT :
        image (bytes) , input_type ("pdf" or "image")
        Output :
        Json ( used by formatting apis )
        """
        try:
            result = {}
            if (self.doc_type).lower() == "math" :
                return math_ocr( self.image_path , self.input_type )

            print("LayoutOCR: Text extraction")
            try:
                ocr_result = image_to_text(self.image_path, self.input_type)
                ocr_result = ocr_result["analyzeResult"]["readResults"]
            except :
                print("LayoutOCR : Text extraction failing , using fallback ocr")
                return fallback_ocr (self.image_path ,self.input_type )
                
            if self.input_type == "image":
                print("LayoutOCR: getting image for further processing")
                img = cv2.imdecode(np.frombuffer(self.image_path, np.uint8), -1)
                img = image_formating(img)
                self.image_path = cv2.imencode(".jpg", img)[1].tostring()
                ocr_text = ocr_result[0]
                result[0] = self.get_result(ocr_text)
            else:
                for page in ocr_result:
                    for line in page['lines']:
                        word_lis=line['text'].split()
                        for idx,word in enumerate(line['words']):
                            temp=dict()
                            temp['boundingBox'] = word
                            temp['text'] = word_lis[idx]
                            line['words'][idx]=temp
                print("LayoutOCR: extracting pages from pdf for further processing")
                images = convert_from_bytes(self.image_path)
                print("LayoutOCR: extraction complete with " + str(len(images)) + " images")

                for index, image in enumerate(images):
                    print("PAGE : ",index+1)
                    ocr_text = ocr_result[index]
                    self.image_path = image_to_byte_array(image)
                    temp = self.get_result(ocr_text)
                    result[index] = temp
                        
            # print(result)
            return result        
        except Exception as ex:
            print("LayoutOCR: error detected in result ",ex)
            return_data = {}
            return_data["error"] = [
                dict(idx="error", content=dict(error="Layout ocr is failing"))
            ]
            return return_data



    def get_result(self, ocr_text):
        """
        Helper function. Which performs the following task:
            1. Layout inference - layout API predictions + fixing title at bbox level
            2. Recreation - Combines ocr text + Layout inference + image to create intermediate json 
            3. json_formating - encode data into more meaningful way
        """
        try:
            bounding_box = LayoutInference(self.image_path).predict
            bounding_box=layout_fixer(bounding_box)
            recreation = ImageRecreation(
                ocr_text=ocr_text, bounding_box=bounding_box, img_file=self.image_path
            )
            json_data = recreation.extract
            doc_object = json_formating(self.image_path, json_data)
            doc_object = post_processing ( doc_object )
            doc_object = line_break_fixer ( doc_object )
            doc_object = list_handling ( doc_object )
            return doc_object
        except Exception as ex:
            print("LayoutOCR: error detected in get_result ",ex)
            return fallback_ocr(self.image_path , "image")
