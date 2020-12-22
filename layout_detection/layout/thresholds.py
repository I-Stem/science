# if the starting x coord lies beyond this then it's multi col
multi_col_threshold = .45
# the height of a box should be atleast this , to contain a text line
minimum_height = .005
# beyond this the list ele becomes multi lined and is probably a proper list
maximum_height_list = .03
# beyond this the text ele becomes multi lined and is probably a proper para
maximum_height_para = .025 
# the distance to which we search for paras up and below for a text boxes
maximum_distance_text = .03
# the distance to which we search for paras up and below in multi col
maximum_distance_text_multi = .02
# the distance to which we search for list up and below 
maximum_distance_list = .03
# the width threshold for converting text to title in multi col
max_width_para_multi = .35
# the width threshold for converting text to title
max_width_para = .65
# the text elements found below and above our ele ( text ) of interest should be atleast this thick
neighbour_height_threshold = .05
# similar for multi col
neighbour_height_threshold_multi = .03