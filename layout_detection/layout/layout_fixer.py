import os
import cv2
import attr
from .thresholds import *
from .post_processing import remove_punc_spaces
import numpy as np
multi_col=False
from copy import deepcopy 

def line_break_fixer ( result ) :
    temp = deepcopy( result )
    try :
        for col in result :
            if col['idx'] == -1 :
                continue
            line_lengths =[]
            for line in col['content'] :
                if line['type']=='p':
                    text =  " ".join([st["content"].strip() for st in line["block"]])
                    line_lengths.append(remove_punc_spaces(text))
            threshold= np.percentile(line_lengths, 10)
            if 1 < threshold < 6  :
                for idx,line in enumerate(col['content']) :
                    if line['type']=='p':
                        text =  " ".join([st["content"].strip() for st in line["block"]])
                        length = remove_punc_spaces(text)
                        if length < threshold :
                            if idx+1<len(col['content']) and line['paragraph_id'] == col['content'][idx+1]['paragraph_id']:
                                    line['alignment'] = 'new_line'
    except Exception as ex:
        print ("Error in line_break_fixer skipping it ",ex)
        result = temp
        
    return result
    
# Checks if alignment "box" lies inside other box
def inside_other ( box, boxes, thresh_x, thresh_y, label_type=None ) :## the thresh are so choosen that it can't contain a new text line where the boxes don't overlap
    for i in boxes :
        if i == box : 
            continue
        if box['bbox'][0] <= i['bbox'][0]+thresh_x and  box['bbox'][1] <= i['bbox'][1]+thresh_y  and  box['bbox'][2] <= i['bbox'][2]+thresh_x and  box['bbox'][3] <= i['bbox'][3]+thresh_y :
            if label_type :
                if i['label'] in label_type : 
                    continue
                return True
    return False

# returns the height of box with specified label which lies above/below it within 'distance'
def search ( boxes, anchor_box, distance, direction, label_type=None ) : 
    global multi_col
    for idx, i in enumerate( boxes ) :
        if i == anchor_box :
             continue
        if multi_col : 
             # if both the boxes lie on same side
            if not ( i['relative_box'][0] > multi_col_threshold ) == ( anchor_box['relative_box'][0] > multi_col_threshold ) :
                continue
        if label_type :
            if not ( i['label']==label_type ) :  
                continue
        if direction == 'up' :
            # if a box lies within 'distance' upwards
            if ( anchor_box['relative_box'][1] - i['relative_box'][3] ) < distance and ( anchor_box['relative_box'][1] - i['relative_box'][3] ) > 0 : 
                return ( i["relative_box"][3] - i["relative_box"][1] )
        else :
            #if a box lies within 'distance' downwards
            if ( i['relative_box'][1] - anchor_box['relative_box'][3] ) < distance and ( i['relative_box'][1] - anchor_box['relative_box'][3] ) > 0 : 
                return ( i["relative_box"][3] - i["relative_box"][1] )

    return False

def layout_fixer ( inp ) :
    temp = deepcopy(inp)
    try:
        global multi_col
        height = inp['page_height'] 
        width = inp['page_width']
        boxes = inp['boxes']
        second_col_ele = 0 # no. of elements that lie on right side

        for box in boxes :
            if box['label'] != 'figure' :
                if box['relative_box'][0] > multi_col_threshold : #if the starting x coord lies on the right side( .45 of width )
                    second_col_ele += 1

        # if no. of elements are > 1 then it is treated as multi col
        if second_col_ele > 1 :
            multi_col = True 

        # handling wrongly classified LIST elements 
        for idx, box in enumerate( boxes ) :
            #if the height is small
            if box['label'] == 'list' and minimum_height< ( box['relative_box'][3] - box['relative_box'][1] ) < maximum_height_list :
                # if current box does not have a list element above and below it within distance -> maximum_distance_list
                if ( not search( boxes, box, maximum_distance_list , "up", 'list' ) ) and ( not search( boxes, box, maximum_distance_list, "down", 'list' ) ) :
                    if not inside_other( box, boxes, height//80, width// 60 , ['text', 'list'] ) :
                        boxes[idx]['label'] = 'title'

        # handling wrongly classified PARA elements 
        for idx, box in enumerate( boxes ) :
            if box['label'] == 'text' :
                # if the height is small
                if not minimum_height < ( box['relative_box'][3] - box['relative_box'][1] ) < maximum_height_para :
                    continue 
                if inside_other( box, boxes, height // 80, width // 60 , ['text', "list"] ) :
                    continue
                if multi_col :
                    #if the width is small 
                    if box['relative_box'][2] - box['relative_box'][0] <  max_width_para_multi :
                        # if text boxes that lie above/below have height -> maximum_distance_text_multi
                        if ( search( boxes, box, maximum_distance_text_multi, "up", 'text' ) > neighbour_height_threshold_multi ) and ( search( boxes, box, maximum_distance_text_multi, "down", 'text' ) > neighbour_height_threshold_multi ) : 
                                boxes[idx]['label'] = 'title'
                # more relaxed thresholds for single col document
                else : 
                    if box['relative_box'][2] - box['relative_box'][0] < max_width_para : #if the width is small 
                    # if text boxes that lie above/below have height -> maximum_distance_text
                        if ( search( boxes, box, maximum_distance_text , "up", 'text' ) > neighbour_height_threshold ) and ( search( boxes, box, maximum_distance_text , "down", 'text' ) > neighbour_height_threshold ) : 
                                boxes[idx]['label'] = 'title'
        output={
            'page_height' : height, 
            'page_width' : width, 
            'boxes' : boxes
        }
    except Exception as ex :
        print( "Error in layout_fixer skipping it ", ex)
        output = temp
    return output
