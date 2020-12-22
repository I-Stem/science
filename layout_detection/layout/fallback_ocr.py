from google.cloud import vision
from google.oauth2.service_account import Credentials
import io
import os
from pdf2image import convert_from_bytes

def detect_document( image_content ,  ocr_credentials ):
    """Detects document features in an image."""

    client = vision.ImageAnnotatorClient( credentials = ocr_credentials )
    image = vision.types.Image( content = image_content )
    response = client.document_text_detection( image = image ) 

    para_text_list = []
    document = response.full_text_annotation
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para_text= ""
                for word in paragraph.words:
                    curr_word=""
                    for symbol in word.symbols:
                        curr_word += symbol.text
                    para_text = para_text+ " " +curr_word
                para_text_list.append( para_text )

    content_list = []
    tmp = dict(
            idx=0,
            paragraph_id= -1 ,
            type="title",
            font_size="h6",
            font_family=None,
            font_color=None,
            alignment=None,
            bold=None,
            italic=None,
            underline=None,
            block=[dict(content="Processed as plain text")],
        )
    content_list.append( tmp )

    for index, i in enumerate( para_text_list ):
        tmp = dict(
            idx=index+1,
            paragraph_id= index+1 ,
            type="p",
            font_size=None,
            font_family=None,
            font_color=None,
            alignment=None,
            bold=None,
            italic=None,
            underline=None,
            block=[dict(content=i)],
        )
        content_list.append( tmp )

    result = []
    result.append( dict( idx= 1 , content =  content_list) )
    result.append( dict( idx=-1, content=dict(full_text = document.text.split("\n")) ) )
    return result 

def fallback_ocr ( image_path ,input_type) :
    ocr_credentials = os.environ['FALLBACK_OCR']
    ocr_credentials = Credentials.from_service_account_file(ocr_credentials)
    result = {}
    print("Fallback_ocr: Text extraction")
    if input_type == "pdf" :
        print("Fallback_OCR: extracting pages from pdf for further processing")
        images = convert_from_bytes(image_path)
        print("Fallback_OCR: extraction complete with " + str(len(images)) + " images")
        for index, image in enumerate(images):
            print("PAGE : ",index+1)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            temp = detect_document ( img_byte_arr , ocr_credentials )
            result[index] = temp
    else :
        result = detect_document( image_path , ocr_credentials )

    return result


