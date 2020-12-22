"""
It is helpful for applying the OCR on the input files.
"""

import os
import time
import requests
from .error_handler import OCRError
from typing import Dict

def _image_to_text(input_file) -> Dict:
    try:
#Image restrictions
        ocr_endpoint = os.getenv("OCR_ENDPOINT")
        headers = {
            os.getenv("OCR_AUTH_KEY1"): os.getenv("OCR_AUTH_VALUE1"),
            os.getenv("OCR_AUTH_KEY2"): os.getenv("OCR_AUTH_VALUE2"),
            "Content-Type": os.getenv("OCR_CONTENT_TYPE")
        }

        print("OCR: sending http request to OCR endpoint: " + ocr_endpoint)
        response = requests.post(ocr_endpoint, headers=headers, data=input_file)
        print("OCR: sent request")
        print("OCR: Received " + str(response.status_code) + " code from model endpoint")
        response.raise_for_status()
        operation_url = response.headers[os.getenv("OCR_CALLBACK_HEADER")]
        analysis = {}
        poll = True
        while poll:
            response_final = requests.get(
                response.headers[os.getenv("OCR_CALLBACK_HEADER")], headers=headers
            )
            analysis = response_final.json()
            time.sleep(1)
            if "recognitionResults" in analysis:
                poll = False
            if "status" in analysis and (
                analysis["status"] == "failed" or analysis["status"] == "succeeded"
            ):
                poll = False
        del input_file
        return analysis
    except Exception as ex:
        print("OCR: Call to OCR endpoint is failing...", ex)
        return "OCR is failing"


def image_to_text(input_file, input_type: str = "image") -> Dict:
    """
    Returns the OCR output
    """
    i = 0
    return_data = None
    for i in range(0,3): # number of retries = 3
        print("OCR: try",str(i+1))
        return_data = _image_to_text(input_file)
        time.sleep(10)
        if isinstance(return_data, dict):
            break
    if input_type == "pdf":
        return_data = ocr_formating(return_data)
    return return_data


def ocr_formating(ocr_data, n=96):
    """
    Helper function for the ocr data.
    For pdf, ocr give the bounding box is in the Inch. This function helps to convert inch into pixels.

    """
    pages = ocr_data["analyzeResult"]["readResults"]
    for page in pages:
        page["width"] = int(page["width"] * n)
        page["height"] = int(page["height"] * n)
        for line in page["lines"]:
            line["boundingBox"] = [int(i * n) for i in line["boundingBox"]]
            line["words"] = [
                [int(i * n) for i in word["boundingBox"]] for word in line["words"]
            ]
    return ocr_data
