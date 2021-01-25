import json,os,re
from copy import deepcopy
from scipy import stats 

roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx', 'xxxi', 'xxxii', 'xxxiii', 'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii', 'xxxix', 'xl', 'xli', 'xlii', 'xliii', 'xliv', 'xlv', 'xlvi', 'xlvii', 'xlviii', 'xlix', 'l', 'li', 'lii', 'liii', 'liv', 'lv', 'lvi', 'lvii', 'lviii', 'lix', 'lx', 'lxi', 'lxii', 'lxiii', 'lxiv', 'lxv', 'lxvi', 'lxvii', 'lxviii', 'lxix', 'lxx', 'lxxi', 'lxxii', 'lxxiii', 'lxxiv', 'lxxv', 'lxxvi', 'lxxvii', 'lxxviii', 'lxxix', 'lxxx', 'lxxxi', 'lxxxii', 'lxxxiii', 'lxxxiv', 'lxxxv', 'lxxxvi', 'lxxxvii', 'lxxxviii', 'lxxxix', 'xc', 'xci', 'xcii', 'xciii', 'xciv', 'xcv', 'xcvi', 'xcvii', 'xcviii', 'xcix']
numbers = [ str(i) for i in list( range(1,100) ) ]
alphabets = list(map(chr, range(97, 123)))
threshold_bullet_width = 5 # also used in last ele width selection

def is_ordered ( word ) : # if bullet is a ordered bullet
    word= word.lower()
    return ( word in roman or word  in alphabets or word in numbers )

def is_unordered (text) : 
    return ( (( len (text) >1 and text[1]==' ' ) or len(text)==1 ) and not text[0].isalnum() )

def ordered_type( word ) : # whether roman / numbere / alphabetical 
    word = word.lower()
    if word in numbers :
        return ( "1" , numbers.index(word)+1 )
    if word in roman :
        return ( "I" , roman.index(word)+1 )
    if word in alphabets :
        return ( "A" , alphabets.index(word)+1 )
    return False
        
def is_valid_ordered ( word, previous_bullet ) : # check if current bullet is the next logical to the last bullet encountered
    word = word.lower()
    previous_bullet = previous_bullet.lower()
    try:
        if previous_bullet in roman :
            if roman [roman.index(previous_bullet) +1] == word :
                return True
        if previous_bullet in numbers :
            if numbers [numbers.index(previous_bullet) +1] == word :
                return True
        if previous_bullet in alphabets :
            if alphabets [alphabets.index(previous_bullet) +1] == word :
                return True
    except : 
        pass
    return False

def ordered_list ( word ,len_chars ) : # returns the "list" dic which is added to json ele
    list_dic = {}
    list_dic['type'] = "ordered"
    list_dic['sub_type'] ,list_dic['count'] = ordered_type( word )
    list_dic['level'] =  len_chars
    return list_dic

def list_handling (result) :
    result_copy = deepcopy(result)
    try :
        for col in result :
            if col['idx'] == -1 : # ocr_result
                continue
            # status lists used as stacks (pop and append)
            list_threshold = []
            list_char = []
            list_para_id = [] 
            list_idx = [] 
            for idx , line in enumerate( col['content'] ) :                       
                while  len (list_idx) > 0  : # non empty lists i.e. bullent points encountered earlier
                    if idx - list_idx [-1] > list_threshold [-1] or line['type'] !='p' or line['paragraph_id'] != list_para_id[-1] : # conditions for resetting lists
                        list_threshold.pop()
                        list_idx.pop()
                        list_para_id.pop()
                        list_char.pop()
                    else : # valid candidate found 
                        break
                if line['type'] != 'p' :
                    continue
                text =  " ".join( [st["content"].strip() for st in line["block"]] )
                try:
                    first_word = re.sub(r'[^\w\s]', '', text).split()[0] # first word with punctuations removed 
                    if first_word == text.split()[0] : # no punctuation mark in first word
                        first_word = ""
                except: # one non alphanumeric char encountered ( split returns error )
                    first_word =''
                
                is_unordered_flag = is_unordered( text )
                is_ordered_flag = is_ordered( first_word )
                if ( is_ordered_flag ) or ( is_unordered_flag ): # bullet detected
                    
                    if len( list_idx ) > 0 : # bullet encountered earlier
                        
                        if ( ( is_unordered_flag and text[0] == list_char[-1] ) or ( is_ordered_flag and is_valid_ordered( first_word,list_char[-1] )) ) : # same bullet point
                            if is_ordered_flag :
                                list_dic_prev = ordered_list ( list_char[-1], len(list_char) )
                                col['content'][list_idx[-1]]['list'] = list_dic_prev # prev occurennce
                                list_dic_current = ordered_list ( first_word , len(list_char) )
                                line['list'] = list_dic_current # current occurence
                                list_char[-1] = first_word
                            else :
                                list_dic = {}
                                list_dic['type'] = "unordered"
                                list_dic['level'] =  len(list_char)
                                col['content'][list_idx[-1]]['list'] = list_dic
                                line['list'] = list_dic

                            for line_ in col['content'][ list_idx[-1] : idx+1 ] :  # marking all para's in b/w as list
                                line_['alignment'] = "list"
                            list_idx[-1] = idx # moving idx to current line. rest lists will be reset if above conditions are not met
                            
                        elif len (list_char) > 1 and ( ( is_unordered_flag and text[0] == list_char[-2] ) or ( is_ordered_flag and is_valid_ordered( first_word,list_char[-2] )) ) : # coming out of nested list
                            # removing elements of nested list
                            list_threshold.pop()
                            list_idx.pop()
                            list_para_id.pop()
                            list_char.pop()

                            if is_ordered_flag :
                                list_dic_prev = ordered_list ( list_char[-1], len(list_char) )
                                col['content'][list_idx[-1]]['list'] = list_dic_prev # prev occurennce
                                list_dic_current = ordered_list ( first_word , len(list_char) )
                                line['list'] = list_dic_current # current occurence
                                list_char[-1] = first_word
                            else :
                                list_dic = {}
                                list_dic['type'] = "unordered"
                                list_dic['level'] =  len(list_char)
                                col['content'][list_idx[-1]]['list'] = list_dic
                                line['list'] = list_dic

                            for line_ in col['content'][ list_idx[-1] : idx+1 ] :  #marking all para's in b/w as list
                                line_['alignment'] = "list"
                            list_idx[-1] = idx # moving idx to current line. rest lists will be reset if above conditions are not met
                        
                        else : #if encountered bullet does not match with any prev bullet
                            if is_ordered_flag and ordered_type(first_word) == ordered_type(list_char[-1]) and first_word.isupper() == list_char[-1].isupper() : # ending prev list and starting new list
                                list_threshold[-1] = threshold_bullet_width
                                list_idx[-1] = idx # moving idx to current line. rest lists will be reset if above conditions are not met
                                list_char[-1] = first_word
                                # list_dic_current = ordered_list ( first_word , len(list_char) )
                                # line['list'] = list_dic_current # current occurence

                            else : # mismatch in bullet point, nested bullet point
                                # pushing into status lists
                                list_threshold.append( threshold_bullet_width )
                                list_para_id.append(line['paragraph_id'])
                                if is_ordered_flag :
                                    list_char.append( first_word )
                                else :
                                    list_char.append( text[0] )
                                list_idx.append( idx )
                    
                    else : # no bullet point was encountered before 
                        # pushing into status lists
                        list_threshold.append( threshold_bullet_width )
                        list_idx.append( idx )
                        list_para_id.append( line['paragraph_id'] )
                        if is_ordered_flag :
                            list_char.append( first_word )
                        else :
                            list_char.append( text[0] )

            #calculating Mode 
            first_line_flag = True
            to_include = False
            first_line_level = -1
            line_widths = []
            temp = []
            for idx , line in enumerate( col['content'] ) :

                if "list" in line and first_line_flag: # to save level of first ele of list
                    first_line_flag = False
                    first_line_level = line['list']['level']

                if "list" in line : # bullet point encountered 
                    if temp!= [] :
                        line_widths.append( len(temp) )
                        temp = []
                    if line['list']['level'] == first_line_level : # calculating widths for first level
                        to_include = True
                    else :
                        to_include = False

                if to_include : # include ele from one bullet point
                    temp.append("1")

                if idx != 0 and "list" in col['content'][idx -1] and line['alignment'] != "list" : # last ele of list has bullet 
                    last_ele_width = 1
                    if len (line_widths) > 2 and col['content'][idx -1]['list']['level'] == first_line_level :
                        last_ele_width = int(stats.mode(line_widths)[0])
                        if last_ele_width > threshold_bullet_width :
                            last_ele_width = threshold_bullet_width
                    #reset
                    first_line_flag = True
                    to_include = False
                    first_line_level = -1
                    line_widths = []
                    temp = []
                    para_id = col [ 'content' ][ idx -1 ][ 'paragraph_id' ] # para id of list
                    for i in range(0,last_ele_width) :
                        if col [ 'content' ][ idx + i ][ 'paragraph_id' ] != line[ 'paragraph_id' ] :
                            break
                        if idx + i < len(col [ 'content' ]) :
                            col [ 'content' ][ idx + i ]['alignment'] = "list"
        return result
        
    except Exception as e:
        print("LIST HANDLING : ERROR. skipping list handling ",e)
        return result_copy