"""
It is helpful for the convert layout json ocr into the mp3 format
"""
import os
import attr
import requests
from xml.etree import ElementTree


@attr.s
class TextToSpeech(object):
    access_token = attr.ib(default=None)

    def __attrs_post_init__(self):
        fetch_token_url = os.getenv("TTS_FETCH_URL")
        headers = {os.getenv("TTS_AUTH_KEY"): os.getenv("TTS_AUTH_VALUE")}
        response = requests.post(fetch_token_url, headers=headers)
        self.access_token = str(response.text)

    def get_audio(self, text):
        constructed_url = os.getenv("TTS_URL")
        headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
        }
        xml_body = ElementTree.Element("speak", version="1.0")
        xml_body.set("{http://www.w3.org/XML/1998/namespace}lang", "en-us")
        voice = ElementTree.SubElement(xml_body, "voice")
        voice.set("{http://www.w3.org/XML/1998/namespace}lang", "en-US")
        voice.set(
            "name", os.getenv("TTS_VOICE")
        )
        voice.text = text
        body = ElementTree.tostring(xml_body)

        response = requests.post(constructed_url, headers=headers, data=body)
        if response.status_code == 200:
            return response.content
        else:
            return b"Facing an Error"


def table_formating(table):
    data = table["block"][0]["content"]
    line_1 = f"table with {len(data)} rows and {len(data[0])} columns .... "
    for index, i in enumerate(data):
        line_1 += f"row {index+1} .... "
        for c in i:
            line_1 += f"{c} .... "
    line_1 += "table end .... "
    return line_1


def audio_convertor(json_response, final_dataset=""):

    try:
        for data in json_response[:2]:
            if data["idx"] == -1:
                continue
            para_index = 0
            para_string = ""
            for index, i in enumerate(data["content"]):
                if i["type"] == "table":
                    final_dataset += table_formating(i)

                if i["type"] == "p":
                    if i["paragraph_id"] == para_index:
                        string = " ".join([st["content"] for st in i["block"]])
                        para_string += string
                    else:
                        if len(para_string.strip()) > 2:
                            final_dataset += para_string + " ...... "
                        para_index = i["paragraph_id"]
                        string = " ".join([st["content"] for st in i["block"]])
                        para_string = string

                if i["type"] == "title":
                    if len(para_string.strip()) > 2:
                        final_dataset += para_string + " ...... "

                    string = " ".join([st["content"] for st in i["block"]])
                    string = " Title ,,,,,," + string + ",,,,,,"
                    final_dataset += string
                    para_string = ""

                if i["type"] == "image":
                    if len(para_string.strip()) > 2:
                        final_dataset += para_string + " ..... "
                    string = i["block"][0]["content"]
                    final_dataset += " ..... There is an image ...... "
                    para_string = ""

            final_dataset += para_string + " ...... "
    except Exception as ex:
        print(ex)
        final_dataset = ""
    return final_dataset


def audio_formating(json_response):
    formating = ""
    for i, j in json_response.items():
        formating = audio_convertor(j, formating)
    tts = TextToSpeech()
    audio = tts.get_audio(formating)
    return audio
