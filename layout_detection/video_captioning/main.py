# Load in the environment variables from .env file
from dotenv import load_dotenv

load_dotenv()
from bson.json_util import dumps
from flask import send_file


# Module imports
import requests
import os
import uuid
from layout_formating.save_file import bucket_saver
import pysrt
import json
from io import StringIO
from isodate import parse_duration
import urllib.request as urllib
import math
from .enum import RequestType, FileExtensions
from zipfile import ZipFile
from os import remove

location = os.environ.get('LOCATION')
account_id = os.environ.get('ACCOUNT_ID')
video_auth_key = os.environ.get('VIDEO_AUTH_KEY')
video_auth_value = os.environ.get('VIDEO_AUTH_VALUE')

# Constants
locale = 'en-us' # locale for custom speech models

def get_account_token():
    url = os.environ.get('VIDEO_GET_ACCOUNTS').format(location)
    params = {'generateAccessTokens' : True, 'allowEdit' : True}
    requestId = str(uuid.uuid4())
    headers = {video_auth_key: video_auth_value, os.environ.get('VIDEO_TELEMETRY_HEADER'): requestId}
    res = requests.get(url=url, params=params, headers=headers)
    token = ''
#    print(dumps(res.json(), indent=2))
    if res.status_code == 200:
        token =  [x for x in res.json() if x['accountType'] == 'Paid'][0]['accessToken']
    else:
        print("fail to obtain account access token: " + res.text)
    print("generated token: " + token)
    return token

def get_token():
    url = os.environ.get('VIDEO_TOKEN').format(
        location, account_id)
    params = {"allowEdit": "true"}
    requestId = str(uuid.uuid4())
    print(requestId)
    headers = {video_auth_key: video_auth_value, os.environ.get('VIDEO_TELEMETRY_HEADER'): requestId}
    res = requests.get(url=url, params=params, headers=headers)
    return res.text[1:-1]

def get_video_duration(video_id):
    url = os.environ.get('VIDEO_GET_ARTIFACT_URL').format(location, account_id, video_id)
    params = {'type': 'Metadata', 'accessToken' : get_account_token()}
    headers = {video_auth_key: video_auth_value}
    res = requests.get(url = url, params = params, headers=headers)
    if res.status_code is 200:
        print(res.text.strip('"'))
        res1 = requests.get(res.text.strip('"'))
        out = res1.json()
        print(dumps(out, indent=2))
        video_seconds = parse_duration(out['AssetFile'][0]['Duration']).total_seconds()
        print("seconds" + str(video_seconds))
        return video_seconds
    return 0

def upload_video(request_data):
    print("initiating video upload process...")
    account_access_token = get_account_token()
    url = os.environ.get('VIDEO_UPLOAD').format(
        location, account_id)
    params = {
        'accessToken': account_access_token,
        'name': request_data.get('name'),
        'language': 'auto', #'en_US',
        'videoUrl': request_data.get('url'),
        'streamingPreset': 'Default',
        'externalId': request_data.get('hash'),
        'privacy': 'Private',
        'callbackUrl': os.environ.get('VIDEO_CALLBACK_SERVER')
    }

    if request_data.get('languageModelId'):
        params['linguisticModelId'] = request_data.get('languageModelId')

    headers = {'Content-Type': 'multipart/form-data', os.environ.get('VIDEO_TELEMETRY_HEADER'): str(uuid.uuid4())}
    res = requests.post(url=url, params=params, headers=headers)

    data = res.json()
    print("response from video API")
    print(data)
    if 'id' in data:
        videoId = data['id']
        return {'error': False, 'message': 'Video Uploaded Successfully', 'videoId': videoId}
    return {'error': True, 'message': 'Error in Uploading Video'}

def train_custom_speech(request_data):
    print("Training custom speech model")
    account_access_token = get_account_token()
    model_name = request_data.get('name')
    url = os.environ.get('VIDEO_CREATE_MODEL').format(
        location, account_id)
    params= {
        'accessToken': account_access_token,
        'modelName': request_data.get('name'),
        'language': locale
    }

    json = {
        request_data.get('fileName'): (None, request_data.get('fileUrl'), None)
    }

    headers = {os.environ.get('VIDEO_TELEMETRY_HEADER'): str(uuid.uuid4())}

    res = requests.post(url = url, params = params, headers = headers, files = json)

    data = res.json()
    print(data)

    if data["files"] and len(data["files"]) > 0:
        model_id = data["id"]
        training_url = os.environ.get('VIDEO_TRAIN_MODEL').format(
            location, account_id, model_id)
        params = {
            'accessToken': account_access_token
        }
        res = requests.put(url = training_url, params = params, headers = headers)

        data = res.json()
        print("response from model training")
        print(data)

        if data['languageModelId'] and len(data['languageModelId']) > 0:
                return {'error': False, 'message': 'Custom speech model successfully trained', 'languageModelId': data['languageModelId']}

    return {'error': True, 'message': 'Error in Uploading Video'}

def video_callback(video_id, req_type, hash, documentName, outputFormat):
    """
        Video Captioning Callback
        ---
        @params: video_id, req_type
    """
    file_url, file_hash = "", hash # S3 File URL, Video File Hash
    if RequestType(req_type) == RequestType.CAPTION:
        file_url, file_hash = get_srt_file(video_id, hash, documentName, outputFormat)
        if file_url is None or len(file_url) == 0:
            print("captioning failed, please implement retry if server problem")
        else:
            print(file_url)

    elif RequestType(req_type) == RequestType.OCR:
        file_url, file_hash = get_text(video_id, hash, documentName)
        if file_url is None or len(file_url) == 0:
            print("OCR failed, please implement retry if server problem")
        else:
            print(file_url)
    
    elif RequestType(req_type) == RequestType.OCR_CAPTION:
        caption_file, ocr_file = None, None
        try:
            caption_file, file_hash = get_srt_file(video_id, hash, documentName, outputFormat, RequestType.OCR_CAPTION)
            if caption_file is None:
                print("captioning failed, please implement retry if server problem")
                caption_file = None
            else:
                print("Captioning successful")
        except:
            print("Caption file error, empty Caption")
        try:
            ocr_file, file_hash = get_text(video_id, hash, documentName, RequestType.OCR_CAPTION)
            if ocr_file is None:
                print("OCR failed, please implement retry if server problem")
                ocr_file = None
            else:
                print("OCR successful")
        except:
            print("OCR file error, empty OCR")
        try:
            if caption_file is not None:
                zip_files(caption_file, file_hash, f'caption-output.{outputFormat}')
            if ocr_file is not None:
                zip_files(ocr_file, file_hash, 'ocr-output.txt')
            file_url = bucket_saver(str(file_hash)+'ocr_caption.zip', file_hash, documentName, FileExtensions.ZIP.value)
            if file_url:
                remove(str(file_hash)+'ocr_caption.zip')
        except:
            return {"error": True, "message": "Zip archiving failed"}


    return {'url':file_url, 'hash': file_hash, 'duration': get_video_duration(video_id)}


def zip_files(file, hash, name: str):
    try:
        print('zip:', name)
        with ZipFile(str(hash)+'ocr_caption.zip', 'a') as zipObj:
            zipObj.writestr(name, file)
    except Exception:
        raise

def format_string(time):
    f = time.split('.')
    if len(f) == 1:
        f.append("000")
    return ','.join(f)

def get_srt_file(video_id, hash, documentName, outputFormat, req_type=None):
    url = os.environ.get('VIDEO_CAPTION').format(location, account_id, video_id)
    params = {"format": f'{outputFormat}', "accessToken" : get_account_token(), "language": "en-US"}
    print(video_id)
    headers = { 
                video_auth_key: video_auth_value}
    res = requests.get(url=url, params=params, headers=headers)
    print("Now printing the response from captioning API")
    print(res.status_code)
    print(res.headers)
    if res.status_code == 200:
        strFile = StringIO(res.text.replace("\r\n", "\n"))
        if req_type is not None:
            return StringIO.read(strFile).encode('utf-8'), hash
        url = bucket_saver(StringIO.read(strFile).encode('utf-8'), hash, documentName, outputFormat)
        strFile.close()
        print(url)
        return url, hash

def get_jaccard_similarity(list1, list2):
    str1List = [elem.split() for elem in list1]
    list1Flattened = [item for sublist in str1List for item in sublist]
    str2List = [elem.split() for elem in list2]
    list2Flattened = [item for sublist in str2List for item in sublist]
    intersection = len(list(set(list1Flattened).intersection(list2Flattened)))
    union = (len(list1Flattened) + len(list2Flattened)) - intersection
    return float(intersection) / union

def get_timestamp_from_seconds(number_of_seconds):
    print(number_of_seconds)
    hours = math.floor(number_of_seconds / 3600.0)
    minutes = math.floor((number_of_seconds - (hours * 3600)) / 60.0)
    seconds = math.floor((number_of_seconds - ((hours * 3600) + (minutes * 60))))
    return str(hours).zfill(2) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2)

def process_extracted_text(ocr_text):
    fileText = "Video text output\n\n"
    parsedJson = json.loads(ocr_text)

    eventResults = []

    timescale = parsedJson['timescale']
    for fragment in parsedJson['fragments']:
        fragmentResults = []
        startTime = get_timestamp_from_seconds(fragment['start'] / timescale)
        endTime = get_timestamp_from_seconds((fragment['start'] + fragment['duration']) / timescale)
        if 'events' in fragment.keys():
            for event in fragment['events']:
                temp = []
                for elem in event:
                    eventText = ""
                    for line in elem['region']['lines']:
                        eventText = eventText + "\n" + line['text']
                    temp.append(eventText)
                if len(eventResults) > 0 and (eventResults[-1] == temp or get_jaccard_similarity(temp, eventResults[-1]) > 0.5):
                    continue
                eventResults.append(temp)
                fragmentResults.append(temp)


        if len(fragmentResults) > 0:
            fileText = fileText + startTime + " to " + endTime

            for sublist in fragmentResults:
                for elem in sublist:
                    fileText = fileText + elem
                fileText = fileText + "\n"
            fileText = fileText + "\n"

    return fileText

def get_text(video_id, hash, documentName, req_type=None):
    url = os.environ.get('VIDEO_GET_ARTIFACT_URL').format(location, account_id, video_id)
    params = {"accessToken": get_account_token(), "language": "en-US", "type": "Ocr"}
    print(video_id)
    headers = { 
                video_auth_key: video_auth_value}
    res = requests.get(url=url, params=params, headers=headers)
    print("Now printing the response from video API")
    print(res.status_code)
    if res.status_code == 200:
        resultUrl = res.text[1:-1]
        print(resultUrl)
        res = requests.get(url=resultUrl, params=params, headers=headers)
        print(res.status_code)
        if res.status_code == 200:
            if "lines" in res.text:
                fileText = process_extracted_text(res.text)
                strFile = StringIO(fileText.replace("\r\n", "\n"))
                if req_type is not None:
                    return StringIO.read(strFile).encode('utf-8'), hash
                url = bucket_saver(StringIO.read(strFile).encode('utf-8'), hash, documentName, 'txt')
                strFile.close()
                print(url + " for text extracted")
                return url, hash
            else:
                ocr_text = 'No text present in the uploaded Video/Audio!\n'+'='*30
                if req_type is not None:
                    return StringIO.read(StringIO(ocr_text)).encode('utf-8'), hash
                url = bucket_saver(StringIO.read(StringIO(ocr_text)).encode('utf-8'), hash, documentName, 'txt')
                return url, hash
    return "", hash