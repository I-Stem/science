from pdf2image import convert_from_bytes
import io
import os
import base64
import requests
import json
from .error_handler import OCRError

app_id = os.getenv("MATH_OCR_APP_ID")
app_key = os.getenv("MATH_OCR_APP_KEY")
ocr_url = os.getenv("MATH_OCR_URL")

def get_result_maths( img ,page_id = 0 ) :
    headers = {'Content-Type': 'application/json', 'app_id': app_id, 'app_key': app_key }
    payload = {
               'src': "data:image/jpeg;base64,"+base64.b64encode(img).decode("utf-8"),
               "formats": ["text", "data", "html"],
               "include_line_data": 'false',
               "data_options": {
                                    "include_table_html": 'true',
                                    "include_mathml":"true"
                                }
                }
    response = requests.post(url = ocr_url, headers = headers, json = payload)

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
                try:
                    result[index] = get_result_maths ( img_byte_arr , index )
                except:
                    result[index] = dict(html="<h6>{Blank page}</h6>" )
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

