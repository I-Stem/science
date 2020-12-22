"""
fixer will help to remove the duplicated data.
Some  time layout analysis detects area as image and if same area is detected as table then it helps us to remove that duplication 

"""

from typing import Dict


def repeating_fixer(l1: Dict) -> dict:
    """
    Handle the conflict/ambiguity betweet image and table detection.

    Parameters
    ----------
    l1 : Dict
        It is column wise data into readable order.


    Return
    ------
        Dict

    Example:
    -------------------
    L1 = dict(text=doc,image_bbox=image_bbox, table_bbox = table_bbox)

    """
    image_data = []
    table_data = []
    for key in sorted(list(l1.keys())):
        i = l1[key]
        img = i["image_bbox"]
        tb = i["table_bbox"]
        d = {}
        for k, v in img.items():
            if v not in image_data:
                d[k] = v
                image_data.append(v)
            else:
                if k in i["text"]:
                    i["text"].remove(k)

        d1 = {}
        for k, v in tb.items():
            if v["box"] not in table_data:
                d1[k] = v
                table_data.append(v["box"])
            else:
                if k in i["text"]:
                    i["text"].remove(k)

        i["image_bbox"] = d
        i["table_bbox"] = d1
        l1[key] = i
    return l1
