from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from layout.layout_ocr import LayoutOcr
from layout_formating.save_file import bucket_saver
from layout_formating.docx_parser import docx_formating
from layout_formating.tts_parser import audio_formating
from layout_formating.html_parser import pdf_formating, text_formating, html_formating
from flask import Flask, flash, request, redirect, url_for, session
from video_captioning.main import upload_video, video_callback, train_custom_speech
from webutils import send_error_response, get_file_content
from PyPDF2 import PdfFileReader
from io import BytesIO

mapper = {"DOCX": "docx", "HTML" : "html", "TXT" : "txt", "PDF" : "pdf", "MP3" :"mp3" }
funct_mapper = {"DOCX":docx_formating, "HTML":html_formating, "TXT":text_formating, "PDF": pdf_formating, "MP3": audio_formating}

mentorship_client = pymongo.MongoClient(os.environ.get('MENTORSHIP_DB_STRING'))
mentorship_db =  mentorship_client['lipdev']['mentorships']

executor = ThreadPoolExecutor(max_workers=20)
app = Flask("layout_detection")
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH')) * 1024 * 1024

@app.route('/api/v1/ocr', methods=['POST'])
def ocr_file():
    (file_content, file_ext) = get_file_content(int(os.environ.get('MAX_OCR_FILE_SIZE')), "ocr")
    hash_value = request.form['hash']
    try :
        doc_type = request.form['doc_type']
    except:
        doc_type = "NONMATH"
    executor.map(get_ocr_result_async, [(file_content, file_ext, hash_value, doc_type )])
    res_msg =  {"message": "Ocr file recorded", "status": "success" }
    return res_msg




def get_ocr_result_async(file_data):
    """ 
    This function will run in background daemon mode.
    doing the OCR conversion.
    """
    input_type= "pdf" if file_data[1] == "pdf" else "image"
    ocr = LayoutOcr(file_data[0], input_type=input_type , doc_type = file_data[3])
    json_data = ocr.result
    url = os.environ.get('ISTEM_OCR_CALLBACK_SERVER')
    print(url)
    input_data = dict()
    if input_type == "pdf":
        pdf_file = PdfFileReader(BytesIO(file_data[0]))
        input_data = {"json": json_data, "hash": file_data[2], "pages": pdf_file.getNumPages()}
    else:
        input_data = {"json": json_data, "hash": file_data[2], "pages": 1}
    print("json request: ")
    # print(input_data)
    result = requests.post(url, json = input_data)
    print("callback sent after ocr completion")



@app.route('/api/v1/ocr/format', methods=['POST'])
def format_data():
    # check if the post request has the file part
    params = request.get_json()
    ocr_output = params['json']
    file_format = params['format']
    file_hash = params['hash']
    document_name = params['documentName']
    return_data = {}
    try:
        if file_format in funct_mapper:
            file = funct_mapper[file_format](ocr_output)
            if file is None:
                return dict(url="", status="fail", message="Formatting Failed", code=500)
            print("sending converted file to aws")
            url = bucket_saver(file, file_hash, document_name, mapper[file_format])
            return_data = dict(url=url,status="success", message="successfully converted", code=200)        
        else:
            message = "please choose valid format"
            return_data = dict(url="", status="fail", message=message, code=200) 
    except Exception as ex:
        print(ex)
        return_data = dict(url="", status="fail", message="Sorry we are facing some issue.", code=500) 
    return return_data



@app.route('/api/v1/vc', methods=['POST'])
def vc_upload():
    params = request.get_json()
    print("vc request received")
    print(params)
    return_data = upload_video(params)
    return json.dumps(return_data)

@app.route('/api/v1/customspeech', methods=['POST'])
def customspeech_train():
    params = request.get_json()
    print("Custom speech training request received")
    print(params)
    return_data = train_custom_speech(params)
    return json.dumps(return_data)

@app.route('/api/v1/vc/callback', methods=['POST'])
def vc_callback():
    print("received callback from backend api")
    params = request.get_json()
    print(params)
    return video_callback(params.get('id'), params.get('type'), params.get('hash'), params.get('documentName'), params.get('outputFormat'))


@app.route('/api/v1/mentorship', methods=['POST'])
def mentorship_matching():
    params = request.get_json()
    try:
        mentee_id = params['request_id']
        cursor = mentorship_db.find()
        database = list(cursor)
        mentee_obj_id = ObjectId(mentee_id)
        mentor_dic = {} # extract latest mentor data
        # date in yyyy/mm/dd format
        threshold_date = datetime(2021, 6, 30)
        for entry in database :
            if entry['_id'] == mentee_obj_id:
                mentee_instance = entry
            if (entry['signupAs'] == 'MENTOR') or (entry['signupAs'] == 'BOTH'): # last entry updated
                if entry['updatedAt'] > threshold_date:
                    mentor_dic[str(entry['userId'])] = entry
                    if entry['mentorshipStatus'][-1]['status'] == "paused":
                        mentor_dic.pop(str(entry['userId']), None)
        mentor_dic.pop(str(mentee_instance['userId']), None)
        mentor_request_id_dataframe = pd.DataFrame.from_records(list(mentor_dic.values())).loc[:, ['_id']]
        for mentor_index in range(mentor_request_id_dataframe.shape[0]):
            mentor_request_id_dataframe.loc[mentor_index, '_id'] = str(mentor_request_id_dataframe.loc[mentor_index, '_id'])
        objective_score_list = objective_score( mentee_instance, list(mentor_dic.values()))
        objective_score_dataframe = pd.DataFrame(objective_score_list, columns=['objective_score', 'objective_reason_for_matching'])

        mentee_skills_type = mentee_skills_type_map[mentee_instance['kindOfSkillsMentee']]

        mentee_dataset_skill_question, mentee_dataset_preference_question, mentee_dataset_miscellaneous_question = generate_mentee_datasets_for_free_form_matching(mentee_instance)
        mentor_dataset_skill_question, mentor_dataset_preference_question, mentor_dataset_miscellaneous_question = generate_mentor_datasets_for_free_form_matching(mentor_dic, mentee_instance)
        doc2vec_model = get_doc2vec_model()
        combined_free_form_match_score_record = get_combined_free_form_match_score_record(doc2vec_model, mentor_dataset_skill_question, mentee_dataset_skill_question, mentee_skills_type, mentor_dataset_preference_question, mentee_dataset_preference_question, mentor_dataset_miscellaneous_question, mentee_dataset_miscellaneous_question)

        mentorship_recommendation_engine_final_match_record = get_mentorship_recommendation_engine_final_match_record(combined_free_form_match_score_record, objective_score_dataframe, mentor_request_id_dataframe)
        top_matches = get_top_matches_mentorship(mentorship_recommendation_engine_final_match_record)

        return_data = dict(matches = top_matches, code = 200)
        return json.dumps(return_data)
    except Exception as ex:
        print("Mentorship_Matching : FAILED ", ex)
        return_data = {}
        return_data["error"] = [
            dict(idx="error", content=dict(error="Mentorship matching is failing"))
        ]
        top_matches = []
        return_data["matches"] = top_matches
        return_data["code"] = 500
        return json.dumps(return_data)

if __name__ == "__main__":
    print('here')
    app.run(port=5000, debug=True, host='0.0.0.0')