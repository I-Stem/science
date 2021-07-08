def get_jaccard_similarity_score(list1 : list, list2 : list):
    """
    
    Parameters
    ----------
    list1 : list
        First list.
    list2 : list
        Second list.

    Returns
    -------
    float
        Jaccard similarity match score between the first and second list.

    """
    try:
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
    except:
        return 0.0
