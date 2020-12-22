import json, re
from nltk.tokenize import word_tokenize
from copy import deepcopy

# where title is marked as separate para
words_threshold = 5 # para should have no more than this many words to label it as title
lines_threshold = 2 # para should have no more than this many lines to label it as title
# in cases where title is appended to last line of prev. para 
words_threshold_last_line = 4 # ,  last line should be smaller than this 
neighbour_word_threshold = 20 # para above and below should have atleast this many words .
lines_threshold_last_line = 4 #  para should have atleast this many lines  
# lines_threshold_neighbour_last_line = 2

ending_puncs = [".", "?", "!"]

# helper function to add line and para meta info ( number of words, number of lines , endswith punctuation )
def adder( para_id, total_words, total_lines, line_info, para_meta_id, para_meta_info, line_meta_info) :
    if para_id != -1 and para_id in para_meta_id :  # if that index is already present
        pos = para_meta_id.index( para_id )
        para_meta_info[pos][0] += total_words
        para_meta_info[pos][1] += total_lines
        line_meta_info[pos][0].extend( line_info )
    else:
        para_meta_id.append( para_id )
        para_meta_info.append( [total_words, total_lines] )
        line_meta_info.append( line_info )


def remove_punc( raw_str ):
    return re.sub(r"[^\w\s]", "", raw_str)


# returns number of words in a sentence after tokenizing and removing punctuations
def remove_punc_spaces( raw_str) :
    tokenized_list = word_tokenize( raw_str )
    processed_str = list( map( remove_punc, tokenized_list ))
    final_str = [word_ for word_ in processed_str if word_ != ""]
    return len( final_str )


# result is a list [0,1,2] which are columns and the entire ocr_text , these list have json with idx ( column number ) and content. content is a list of line/title/img/table jsons
def post_processing_title( result ):
    for idx, col in enumerate( result[:-1] ):  # excluding the last ele which has the ocr_text
        para_meta_id = []  # stores all the paragraph ids in order
        para_meta_info = []  # corresponding to each paragraph id stores - [ total number of words, number of lines ]
        line_meta_info = [] # for all lines in a para has [number of words in that line, does it end with a punctuation , line idx ]
        prev_para_id = -100
        total_words = 0
        total_lines = 0
        line_info = []
        for ele in col["content"] :
            curr_words = 0
            total_lines += 1
            end_punc = False
            if ele["type"] != "p" :  # table, title or figure
                curr_words = -1
            else:
                for cont in ele["block"] :
                    curr_words += remove_punc_spaces( cont["content"] )  #  Tokenize and remove punctuations
                if cont["content"].strip()[-1] in ending_puncs:
                    end_punc = True

            if prev_para_id != int(ele["paragraph_id"]) or prev_para_id == -1:  # if prev_id is not equal to current id or previous id was -1
                adder( prev_para_id, total_words, total_lines, line_info, para_meta_id,  para_meta_info, line_meta_info )
                total_words = 0
                line_info = []
                total_lines = 0
                prev_para_id = int( ele["paragraph_id"] )

            total_words += curr_words  # added after checking if it's same prev_para_id (else it's initialised to zero)
            line_info.append( [curr_words, end_punc, ele["idx"] ] )

        total_lines += 1
        adder( prev_para_id, total_words, total_lines, line_info, para_meta_id,  para_meta_info, line_meta_info )


        # the first ele is the -100 dummy id we inserted
        para_meta_id = para_meta_id [1:]
        para_meta_info = para_meta_info [1:]
        line_meta_info = line_meta_info [1:]

        title_lis = []
        #  IF Bounding BOX FOR TITLE IS DRAWN but marked as "p"
        for index, para_id in enumerate(para_meta_id):
            if para_id != -1 and ( para_meta_info[index][0] <= words_threshold )  and  para_meta_info[index][1] <= lines_threshold :
                if  index + 1 < len( para_meta_id )  and para_meta_id[index + 1] != -1 and para_meta_info[index + 1][0] >= neighbour_word_threshold :  # Para below should have more words
                    if index == 0 or para_meta_id[index - 1] != -1  and para_meta_info[index - 1][0] >= neighbour_word_threshold :  # if its not the first ele then we check the para above it also
                        para_meta_id[index] = -1
                        title_lis.append( para_id )

        if title_lis:
            for ele in result[idx] ["content"]:
                if ele["paragraph_id"] in title_lis:
                    ele["paragraph_id"] = -1
                    ele["type"] = "title"

        line_lis = []
        # IF Bounding BOX FOR TITLE IS NOT DRAWN,
        for index, para_id in enumerate(para_meta_id):
            # if it's not a para OR it's the last element OR the next element is a title THEN we skip that element
            if para_id == -1 or index + 1 == len(para_meta_id) or para_meta_id[index + 1] == -1 :
                continue
            num_lines = para_meta_info[index][1]
            if num_lines > lines_threshold_last_line :  # it's len is above the lines_threshold_last_line
                if line_meta_info[index][num_lines - 2][1] :  # penultimate line ends in punc
                    if not line_meta_info[index][num_lines - 1][1]:  # last line does not end in punc
                        if line_meta_info[index][num_lines - 1][0] <= words_threshold_last_line:  # if that line has less words
                            #  if  para_meta_info[index+1][1] > lines_threshold_neighbour_last_line : #checking if ele below it has more lines
                            line_lis.append(line_meta_info[index][num_lines - 1][2])

        if line_lis:
            for ele in result [idx]["content"]:
                if ele["idx"] in line_lis:
                    ele["paragraph_id"] = -1
                    ele["type"] = "title"

    return result


def shift_para_ids( ele_list ):
    for ele in ele_list:
        if ele["paragraph_id"] != -1:
            ele["paragraph_id"] += 1

 # if multiple para's and title are merged into one 
def post_processing_merged( result ):
    for idx, col in enumerate( result[:-1] ):  # excluding the ocr_text which is in last ele
        prev_punc = False
        prev_len = 0
        for ele_id, ele in enumerate( col["content"][1:-1] ):
            if ele["type"] != "p":
                continue
            curr_len = 0
            curr_punc = False
            for cont in ele["block"]:
                curr_len += remove_punc_spaces( cont["content"] )  # Tokenize and remove punctuations
            if cont["content"].strip()[-1] in ending_puncs:
                curr_punc = True
            next_ele = col["content"][1:][ele_id + 1]
            if curr_len != 0 and curr_len <= 0.3 * prev_len and prev_punc  and (not curr_punc):  # and next_ele["paragraph_id"] == ele["paragraph_id"] :
                next_len = 0
                for cont in next_ele["block"]:
                    next_len += remove_punc_spaces( cont["content"] )
                if curr_len <= 0.3 * next_len:
                    ele["paragraph_id"] = -1
                    ele["type"] = "title"
                    if col["content"][1:-1][ele_id - 1]["paragraph_id"] == next_ele["paragraph_id"]:
                        shift_para_ids(col["content"][1:][ele_id + 1 :])
            prev_punc = curr_punc
            prev_len = curr_len
    return result


def post_processing(doc_obj):
    temp = deepcopy( doc_obj )
    try :
        doc_obj = post_processing_title( doc_obj )
    except Exception as ex:
        print("Error in post_processing_title function, skipping it ",ex)
        doc_obj = temp

    temp = deepcopy( doc_obj )
    try :
        doc_obj = post_processing_merged( doc_obj )
    except Exception as ex:
        print("Error in post_processing_merged function, skipping it ",ex)
        doc_obj = temp
        
    return doc_obj
