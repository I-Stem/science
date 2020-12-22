import os
from flask import  request, jsonify, make_response
from flask_restx import abort
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set([ 'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff' ])

def send_error_response(status_code:int, message:str, status:str="failure"):
    abort(make_response(jsonify(message=message, status=status), status_code))

def get_file_content(file_size:int, flow:str):
    cl = request.content_length
    if cl is not None and cl > file_size * 1024 * 1024:
        send_error_response(413, message="Input file size is too large, the maximum file limit is " + str(file_size) + " MB")
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename == '':
        send_error_response(400, message="The request is missing the file input.")
    file_ext = os.path.splitext(filename)[1].lower()[1:]
    if flow == "ocr" and file_ext not in ALLOWED_EXTENSIONS:
        send_error_response(400, message="Unsupported media file type, file type should be one of: " + " ".join(ALLOWED_EXTENSIONS))
    return file.read(), file_ext