from pdf2image import convert_from_bytes
import io
import os
import base64
import requests
import json
from .error_handler import OCRError

ocr_ENDPOINT = os.getenv("MATH_OCR_ENDPOINT") #The endpoint of the deployed math model

def get_result_maths( img ,page_id = 0 ) :
    headers = {'Content-Type': 'application/json', os.getenv("MATH_OCR_AUTH_KEY1"): os.getenv("MATH_OCR_AUTH_VALUE1"), os.getenv("MATH_OCR_AUTH_KEY2"): os.getenv("MATH_OCR_AUTH_VALUE2")}
    response = requests.post(url = ocr_endpoint, headers = headers, json = json.loads(os.getenv("MATH_OCR_FORMAT")))

    if "error" in response.json() :
        print("MATH_OCR : Error in page {} , error - {}".format( page_id +1 ,response.json()['error_info']))
        raise OCRError        
    return response.json()

def math_ocr ( input_file, input_type: str = "image") :
    try :
        print("MATH_OCR : Text extraction")
        result = {}
        if input_type == "image":
            print("MATH OCR: getting image for further processing")
            result[0] = get_result_maths ( input_file )
        else:
            print("MATH_OCR: extracting pages from pdf for further processing")
            images = convert_from_bytes( input_file )
            print("MATH_OCR: extraction complete with " + str(len(images)) + " images")

            for index, image in enumerate(images):
                print("PAGE : ",index+1)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                result[index] = get_result_maths ( img_byte_arr , index )
            
        result['math'] = "True"
        # print(result)
        return result
    except Exception as e:
        print("ERROR in Math ocr", e)  
        return_data = {}
        return_data["error"] = [
            dict(idx="error", content=dict(error="Math ocr is failing"))
        ]
        return return_data

