import re
import pdfkit
import html2text
import string as string_library
import enchant
from .utils import *

def paragraph_formating(string_data, index):
    string_data = string_data.strip()
    string = f"<p id={index}>{string_data}</p>"
    return string


def title_formating(string_data, index):
    string_data =  string_data.strip()
    if index < 5:
        string = f"<title>{string_data}</title>"
    string = f"<h1>{string_data}</h1>" # title correction
    return string

# meta titles - Page number and column number
def meta_titles_formatting(string_data):
    string_data =  string_data.strip()
    string = f"<h6>{string_data}</h6>" 
    return string

def image_formating(data_uri):
    html = '<img src="data:image/png;base64,{0}" alt="Graphic" >'.format(data_uri)
    return html

def table_formating(temp):
    string = "<table>"
    for data in temp["block"][0]["content"]:
        string += "<tr>"
        for _ in data:
            string += f"<td>{_}</td>"
        string += "</tr>"
    string += "</table></br></br>"
    return string

def list_formatting ( line_list, level=1 ) :
    if line_list[0]['list']['type'] == 'unordered' :
        string = "<ul>"
        end_char = "</ul>"
        unordered_flag= True
    else :
        string = "<ol type={} start= {}>".format ( line_list[0]['list']['sub_type'] , line_list[0]['list']['count'] )
        end_char = "</ol>"
        unordered_flag= False
    bullet_string = ''
    idx = 0
    while ( idx < len(line_list) ) :
        line = line_list[idx]
        line_text =  " ".join( [st["content"].strip() for st in line[ "block" ] ] )
        if 'list' in line:
            line_text = bullet_remover( line_text, unordered_flag )
        if 'list' in line and bullet_string != "" : # new list point encountered
            string += "<li>{}</li>".format(bullet_string)
            bullet_string = ""

        if "list" in line and int ( line['list']['level'] ) == int (level+1 ): # entering nested
            line_text = ""
            new_idx,temp_string = list_formatting( line_list[ idx: ] , ( level +1 ) )
            idx = idx + new_idx
            string += temp_string
            
        bullet_string = bullet_string + " " + line_text

        if "list" in line and line['list']['level'] == level-1 : # exiting nested
            string += end_char
            return idx,string
        
        idx += 1
    if bullet_string != "" :
        string += "<li>{}</li>".format(bullet_string)
    string += end_char
    return idx , string

def _html_formating(json_response, page_id=0):
    if len(json_response) == 1:
        string_data = f"<h1>{json_response[0]['content']['error']}</h1>"
        return _

    if not isinstance(json_response, list):
        string_data = "Input does not match the expected format"
        print("error: " + string_data)
        return string_data

    word_dic_US = enchant.Dict("en_US")
    word_dic_UK = enchant.Dict("en_UK")

    try:
        string_data = meta_titles_formatting(f"Page {str(page_id+1)}")
        if len(json_response) > 2 :
            two_col = True
        else :
            two_col = False
        text_split_flag = False # if word was split in two text lines with "-"

        for data in json_response[:2]:
            if data["idx"] == -1:
                continue
            para_index = 0
            para_string = ""
            string_data += f'<span idx = {data["idx"]}>'
            if two_col :
                string_data += meta_titles_formatting(f'Column {str(data["idx"])}')
            line_list=[]

            for index, i in enumerate( data["content"] ):
                if ( i['alignment'] != None and i['alignment']=='list') :
                    line_list.append(i)
                    continue
                if line_list != []:
                    if len(para_string.strip()) > 2:
                        string_data += paragraph_formating(para_string, para_index)
                    string_data += list_formatting( line_list )[1]
                    line_list = []
                    para_string = ""
                    
                if i["type"] == "table": #position correction
                    if len(para_string.strip()) > 2:
                        string_data += paragraph_formating(para_string, para_index)
                    string_data += table_formating(i)
                    para_string = ""

                if i["type"] == "p":
                    if i["paragraph_id"] == para_index:
                        string = " ".join([st["content"].strip() for st in i["block"]])
                        if i['alignment'] == 'new_line':
                            string = "<br>" + string +"<br>"

                        if text_split_flag : # the last ele had a word split 
                            para_string = para_string[:-1] + string # removing "-" and concatenating two lines.
                            text_split_flag = False
                        else:
                            para_string = para_string + " " +string # space between lines of a para
                    else:
                        if len(para_string.strip()) > 2:
                            string_data += paragraph_formating(para_string, para_index)
                        para_index = i["paragraph_id"]
                        string = " ".join([st["content"].strip() for st in i["block"]])
                        para_string = string

                    if string.endswith("-") : # handling words splited by line break ( separated by "-")
                        if index+1 < len( data["content"] ) and data["content"][index+1]["type"]== "p" and data["content"][index+1]["paragraph_id"] == para_index : # next ele belongs to same para
                            next_line =  " ".join([st["content"].strip() for st in data["content"][index+1]["block"]])
                            try:
                                next_word = next_line.strip()[ : next_line.strip().index(" ")] # word portion that was pushed to next line ( text after '-' ) (first word of next line)
                            except:
                                next_word= next_line # single word 
                            try:
                                prev_word = string[ string.rindex(" ")+1: ] # first part of the word along with '-' (last word of current line)
                            except :
                                prev_word = string # single word 

                            word_splitted = ( prev_word[:-1] + next_word ).lower()
                            if word_splitted[-1] in string_library.punctuation :
                                word_splitted = word_splitted[:-1].lower()
                            if word_dic_US.check( word_splitted ) or  word_dic_UK.check( word_splitted ) : # if after removal of "-" a meaningful word is formed
                                text_split_flag = True
            

                if i["type"] == "title":
                    if len(para_string.strip()) > 2:
                        string_data += paragraph_formating(para_string, para_index)
                    string = " ".join([st["content"].strip() for st in i["block"]])
                    if i['font_size'] == "h6" :
                        string_data += meta_titles_formatting(string)
                    else :
                        string_data += title_formating(string, index)
                    para_string = ""

                if i["type"] == "image":
                    if len(para_string.strip()) > 2:
                        string_data += paragraph_formating(para_string, para_index)
                    string = i["block"][0]["content"]
                    string_data += image_formating(string)
                    para_string = ""
            string_data += paragraph_formating(para_string, para_index)
            if line_list != []:
                string_data += list_formatting( line_list )[1]
            string_data += "</span>"
        return string_data
        
    except Exception as ex:
        print("HTML PARSER: FAILING ,Page",page_id+1,ex)
        string_data = "<h6>HTML RECONSTRUCTION FAILING</h6>" +"<br>"+"<h6>Raw Text -</h6>" + "<br> ".join( json_response[ -1 ][ "content" ][ "full_text" ] )
        return string_data


def html_formating(json_response):
    """
    Layout ocr into html format
    """
    style_data = """<style>
        table, th, td {
          border: 1px solid black;
          border-collapse: collapse;
        }
        </style>"""

    if "math" in json_response :
        string_data = html_mathjax_header
        for page_number, page_content in json_response.items():
            if page_number == "math" : continue
            string_data += meta_titles_formatting(f"Page {str( int(page_number) +1)}")
            string_data += page_content['html'].replace("\"display: none\"" ,"\"display: block\"")
        return string_data + html_mathjax_footer

    string_data = ""
    for page_number, page_content in json_response.items():
        string_data += _html_formating(page_content, int(page_number))
    return f"<html><head>{style_data}</head><body>{string_data}<body></html>"


def text_formating(json_response):
    """
    Layout ocr into text format
    """

    html = html_formating(json_response)
    clean_data = re.sub(
        "(<img.*?>)",
        "[Image]",
        html,
        0,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    text_maker = html2text.HTML2Text()
    text_maker.body_width = 0
    text_maker.pad_tables= True
    text = text_maker.handle(clean_data)
    return text#.replace("|", "\t")

def pdf_formating(json_response):
    """
    Layout ocr into pdf format
    """

    html = html_formating(json_response)
    print("generated html format")
    data = pdfkit.from_string(html, output_path=False)
    return data
