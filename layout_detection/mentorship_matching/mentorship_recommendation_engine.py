import numpy as np
import pandas as pd
import json
import os
from .mm_match_data_preprocessing.nlp_preprocessing import get_nlp_preprocessed_dataset
from .matching import get_match_based_on_cosine_similarity, get_match_based_on_key_phrase_similarity, get_match_based_on_soft_skill_classifier, get_match_based_on_hard_skill_classifier
from dotenv import load_dotenv
load_dotenv()

KEY_PHRASE_SIMILARITY_THRESHOLD = 0.5
COSINE_SIMILARITY_THRESHOLD = 0.4
SOFT_SKILL_SIMILARITY_THRESHOLD = 0.4
HARD_SKILL_SIMILARITY_THRESHOLD = 0.4

SKILLS_QUESTION_THRESHOLD = 0.5
PREFERENCES_QUESTION_THRESHOLD = 0.5

#mentee_dataset = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentee_dataset.xlsx")
#mentor_dataset = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentor_dataset.xlsx")

def get_combined_free_form_match_score_record(doc2vec_model, mentor_dataset_skill_question, mentee_dataset_skill_question, mentee_skills_type : str, mentor_dataset_preference_question, mentee_dataset_preference_question, mentor_dataset_miscellaneous_question, mentee_dataset_miscellaneous_question):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    mentor_dataset_skill_question : pd.DataFrame
        Dataframe containing mentor ids and their text responses for the skills question.
    mentee_dataset_skill_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the skills question.
    mentee_skills_type : str
        soft_skills or hard_skills or soft_and_hard_skills.
    mentor_dataset_preference_question : pd.DataFrame
        Dataframe containing mentor ids and their text responses for the preferences question.
    mentee_dataset_preference_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the preferences question.
    mentor_dataset_miscellaneous_question : pd.DataFrame
        Dataframe containing mentor ids and their text responses for the miscellaneous question.
    mentee_dataset_miscellaneous_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the miscellaneous question.

    Returns
    -------
    combined_free_form_match_score_record : pd.DataFrame
        Dataframe containing matches which are represented using the mentor and mentee ids
        and their corresponding match scores for all the matching algorithms used for freeform text.
        It also contains the overall match score for individual questions as well as the combined freeform text matching score.
        The reason for match is also present in the Dataframe.

    """
    nlp_preprocessed_mentor_dataset_skill_question = get_nlp_preprocessed_dataset(mentor_dataset_skill_question, "mentor")
    nlp_preprocessed_mentee_dataset_skill_question = get_nlp_preprocessed_dataset(mentee_dataset_skill_question, "mentee")
    
    nlp_preprocessed_mentor_dataset_preference_question = get_nlp_preprocessed_dataset(mentor_dataset_preference_question, "mentor")
    nlp_preprocessed_mentee_dataset_preference_question = get_nlp_preprocessed_dataset(mentee_dataset_preference_question, "mentee")
    
    nlp_preprocessed_mentor_dataset_miscellaneous_question = get_nlp_preprocessed_dataset(mentor_dataset_miscellaneous_question, "mentor")
    nlp_preprocessed_mentee_dataset_miscellaneous_question = get_nlp_preprocessed_dataset(mentee_dataset_miscellaneous_question, "mentee")
    
    combined_match_score_record_free_form_skill_question = get_free_form_skill_question_matches(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question, mentee_skills_type)
    combined_match_score_record_free_form_preference_question = get_free_form_non_skill_question_matches(doc2vec_model, nlp_preprocessed_mentor_dataset_preference_question, nlp_preprocessed_mentee_dataset_preference_question)
    combined_match_score_record_free_form_miscellaneous_question = get_free_form_non_skill_question_matches(doc2vec_model, nlp_preprocessed_mentor_dataset_miscellaneous_question, nlp_preprocessed_mentee_dataset_miscellaneous_question)
    
    combined_free_form_match_score_record = combine_match_score_records(combined_match_score_record_free_form_skill_question, combined_match_score_record_free_form_preference_question, combined_match_score_record_free_form_miscellaneous_question)
    
    if (mentee_dataset_preference_question.loc[0, 'mentee_data'] == '') and (mentee_dataset_miscellaneous_question.loc[0, 'mentee_data'] == ''):
        combined_free_form_match_score_record['combined_free_form_match_score'] = combined_free_form_match_score_record['skill_question_overall_match_score']
        combined_free_form_match_score_record = add_reasons_for_matching(combined_free_form_match_score_record, ignore_prefernce_match=True)
        
    elif (mentee_dataset_preference_question.loc[0, 'mentee_data'] == '') and (mentee_dataset_miscellaneous_question.loc[0, 'mentee_data'] != ''):
        combined_free_form_match_score_record['combined_free_form_match_score'] = 0.8*combined_free_form_match_score_record['skill_question_overall_match_score'] + 0.2*combined_free_form_match_score_record['miscellaneous_question_overall_match_score']
        combined_free_form_match_score_record = add_reasons_for_matching(combined_free_form_match_score_record, ignore_prefernce_match=True)
        
    elif (mentee_dataset_miscellaneous_question.loc[0, 'mentee_data'] == '') and (mentee_dataset_preference_question.loc[0, 'mentee_data'] != ''):
        combined_free_form_match_score_record['combined_free_form_match_score'] = 0.7*combined_free_form_match_score_record['skill_question_overall_match_score'] + 0.3*combined_free_form_match_score_record['preference_question_overall_match_score']
        combined_free_form_match_score_record = add_reasons_for_matching(combined_free_form_match_score_record)
        
    else:
        combined_free_form_match_score_record['combined_free_form_match_score'] = 0.6*combined_free_form_match_score_record['skill_question_overall_match_score'] + 0.2*combined_free_form_match_score_record['preference_question_overall_match_score'] + 0.2*combined_free_form_match_score_record['miscellaneous_question_overall_match_score']
        combined_free_form_match_score_record = add_reasons_for_matching(combined_free_form_match_score_record)
        
    return combined_free_form_match_score_record
    
def combine_match_score_records(combined_match_score_record_free_form_skill_question, combined_match_score_record_free_form_preference_question, combined_match_score_record_free_form_miscellaneous_question):
    """
    
    Parameters
    ----------
    combined_match_score_record_free_form_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms used for the skills question.
    combined_match_score_record_free_form_preference_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms used for the preferences question.
    combined_match_score_record_free_form_miscellaneous_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms used for the miscellaneous question.

    Returns
    -------
    combined_free_form_match_score_record : pd.DataFrame
        Dataframe resulting after combining the combined_match_score_record for all the free form questions.

    """
    combined_free_form_match_score_record_ids_part = combined_match_score_record_free_form_skill_question.loc[:, ['mentee_id', 'mentor_id']]
    
    combined_free_form_match_score_record_skill_question_part = rename_columns_by_adding_string_at_start(combined_match_score_record_free_form_skill_question, "skill_question_").loc[:, ['skill_question_match_score_key_phrase_similarity', 'skill_question_match_score_cosine_similarity', 'skill_question_match_score_soft_skill_matching', 'skill_question_match_score_hard_skill_matching', 'skill_question_overall_match_score']]
    
    combined_free_form_match_score_record_preference_question_part = rename_columns_by_adding_string_at_start(combined_match_score_record_free_form_preference_question, "preference_question_").loc[:, ['preference_question_match_score_key_phrase_similarity','preference_question_match_score_cosine_similarity', 'preference_question_overall_match_score']]
    
    combined_free_form_match_score_record_miscellaneous_question_part = rename_columns_by_adding_string_at_start(combined_match_score_record_free_form_miscellaneous_question, "miscellaneous_question_").loc[:, ['miscellaneous_question_match_score_key_phrase_similarity','miscellaneous_question_match_score_cosine_similarity', 'miscellaneous_question_overall_match_score']]
    
    combined_free_form_match_score_record = pd.concat([combined_free_form_match_score_record_ids_part, combined_free_form_match_score_record_skill_question_part, combined_free_form_match_score_record_preference_question_part, combined_free_form_match_score_record_miscellaneous_question_part], axis=1)
    return combined_free_form_match_score_record

def rename_columns_by_adding_string_at_start(DataFrame, string : str):
    """
    
    Parameters
    ----------
    DataFrame : pd.DataFrame
        Dataframe for which the columns are to be renamed.
    string : str
        String to add at the start of the column names to generate new column names.

    Returns
    -------
    DataFrame_with_renamed_columns : pd.DataFrame
        Dataframe generated as output after renaming the columns of the input Dataframe.

    """
    original_column_names_list = list(DataFrame.columns)
    strings_to_add_list = [string for index in range(len(original_column_names_list))]
    new_column_names_list = [strings_to_add_list[index] + original_column_names_list[index] for index in range(len(original_column_names_list))]
    
    new_column_names_dict = dict()
    for index in range(len(original_column_names_list)):
        new_column_names_dict[original_column_names_list[index]] = new_column_names_list[index]
    
    DataFrame_with_renamed_columns = DataFrame.rename(columns = new_column_names_dict)
    return DataFrame_with_renamed_columns

def add_reasons_for_matching(combined_free_form_match_score_record, ignore_prefernce_match=False):
    """
    
    Parameters
    ----------
    combined_free_form_match_score_record : pd.DataFrame
        Dataframe containing matches which are represented using the mentor and mentee ids
        and their corresponding match scores for all the matching algorithms used for freeform text.
        It also contains the overall match score for individual questions as well as the combined freeform text matching score.
    ignore_prefernce_match : bool, optional
        Boolean to decide whether to ignore the preference matching question. The default is False.

    Returns
    -------
    combined_free_form_match_score_record : pd.DataFrame
        Dataframe resulting from addition of reason_for_matching column into input Dataframe.

    """
    reason_for_matching_list = []
    for index in range(combined_free_form_match_score_record.shape[0]):
        reason_for_matching_list.append("")
        if combined_free_form_match_score_record.loc[index, 'skill_question_overall_match_score'] >= SKILLS_QUESTION_THRESHOLD:
            reason_for_matching_list[index] += "Freeform Data : Skillset match "
        if (combined_free_form_match_score_record.loc[index, 'preference_question_overall_match_score'] >= PREFERENCES_QUESTION_THRESHOLD) and (not ignore_prefernce_match):
            reason_for_matching_list[index] += "Freeform Data : Preferences match "
    combined_free_form_match_score_record['free_form_reason_for_matching'] = pd.Series(reason_for_matching_list)
    return combined_free_form_match_score_record

                                                     
def get_free_form_skill_question_matches(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question, mentee_skills_type : str):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_mentor_dataset_skill_question : pd.DataFrame
        Dataframe containing mentor ids and corresponding text mentor_data for the skill question preprocessed for nlp.
    nlp_preprocessed_mentee_dataset_skill_question : pd.DataFrame
        Dataframe containing mentee ids and corresponding text mentee_data for the skill question preprocessed for nlp.
    mentee_skills_type : str
        soft_skills or hard_skills or soft_and_hard_skills.

    Returns
    -------
    combined_match_score_record_free_form_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms used for the skills question.

    """
    if mentee_skills_type == 'soft_skills':
        match_score_record_classifier_soft_skills = get_match_based_on_soft_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_cosine_similarity = get_match_based_on_cosine_similarity(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_key_phrase_similarity = get_match_based_on_key_phrase_similarity(nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        free_form_skill_question_overall_match_score_list = []
        for match_index in range(match_score_record_key_phrase_similarity.shape[0]):
            if match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity'] >= KEY_PHRASE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.75 + match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity']/4)
            elif match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity'] >= COSINE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.5 + match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity']/4)
            else:
                free_form_skill_question_overall_match_score_list.append(match_score_record_classifier_soft_skills.loc[match_index]['match_score_soft_skill_matching']/2)
        combined_match_score_record_free_form_skill_question = get_combined_match_score_record_free_form_skill_question_soft_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_soft_skills, free_form_skill_question_overall_match_score_list)
    elif mentee_skills_type == 'hard_skills':
        match_score_record_classifier_hard_skills = get_match_based_on_hard_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_cosine_similarity = get_match_based_on_cosine_similarity(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_key_phrase_similarity = get_match_based_on_key_phrase_similarity(nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        free_form_skill_question_overall_match_score_list = []
        for match_index in range(match_score_record_key_phrase_similarity.shape[0]):
            if match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity'] >= KEY_PHRASE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.75 + match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity']/4)
            elif match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity'] >= COSINE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.5 + match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity']/4)
            else:
                free_form_skill_question_overall_match_score_list.append(match_score_record_classifier_hard_skills.loc[match_index]['match_score_hard_skill_matching']/2)
        combined_match_score_record_free_form_skill_question = get_combined_match_score_record_free_form_skill_question_hard_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_hard_skills, free_form_skill_question_overall_match_score_list)
    elif mentee_skills_type == 'soft_and_hard_skills':
        match_score_record_classifier_soft_skills = get_match_based_on_soft_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_classifier_hard_skills = get_match_based_on_hard_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_cosine_similarity = get_match_based_on_cosine_similarity(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        match_score_record_key_phrase_similarity = get_match_based_on_key_phrase_similarity(nlp_preprocessed_mentor_dataset_skill_question, nlp_preprocessed_mentee_dataset_skill_question)
        free_form_skill_question_overall_match_score_list = []
        for match_index in range(match_score_record_key_phrase_similarity.shape[0]):
            if match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity'] >= KEY_PHRASE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.75 + match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity']/4)
            elif match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity'] >= COSINE_SIMILARITY_THRESHOLD:
                free_form_skill_question_overall_match_score_list.append(0.5 + match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity']/4)
            else:
                free_form_skill_question_overall_match_score_list.append(match_score_record_classifier_soft_skills.loc[match_index]['match_score_soft_skill_matching']/4 + match_score_record_classifier_hard_skills.loc[match_index]['match_score_hard_skill_matching']/4)
        combined_match_score_record_free_form_skill_question = get_combined_match_score_record_free_form_skill_question_soft_and_hard_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_soft_skills, match_score_record_classifier_hard_skills, free_form_skill_question_overall_match_score_list)
    else:
        print("Error : Incorrect skills_type")
        return None
    return combined_match_score_record_free_form_skill_question
    
def get_free_form_non_skill_question_matches(doc2vec_model, nlp_preprocessed_mentor_dataset_preference_question, nlp_preprocessed_mentee_dataset_preference_question):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_mentor_dataset_non_skill_question : pd.DataFrame
        Dataframe containing mentor ids and corresponding text mentor_data for the non-skill question preprocessed for nlp.
    nlp_preprocessed_mentee_dataset_non_skill_question : pd.DataFrame
        Dataframe containing mentee ids and corresponding text mentee_data for the non-skill question preprocessed for nlp.

    Returns
    -------
    combined_match_score_record_free_form_non_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms used for the non-skill question.

    """
    match_score_record_key_phrase_similarity = get_match_based_on_key_phrase_similarity(nlp_preprocessed_mentor_dataset_preference_question, nlp_preprocessed_mentee_dataset_preference_question)
    match_score_record_cosine_similarity = get_match_based_on_cosine_similarity(doc2vec_model, nlp_preprocessed_mentor_dataset_preference_question, nlp_preprocessed_mentee_dataset_preference_question)
    free_form_non_skill_question_overall_match_score_list = []
    for match_index in range(match_score_record_key_phrase_similarity.shape[0]):
        if match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity'] >= KEY_PHRASE_SIMILARITY_THRESHOLD:
            free_form_non_skill_question_overall_match_score_list.append(0.5 + match_score_record_key_phrase_similarity.loc[match_index, 'match_score_key_phrase_similarity']/2)
        elif match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity'] >= COSINE_SIMILARITY_THRESHOLD:
            free_form_non_skill_question_overall_match_score_list.append(match_score_record_cosine_similarity.loc[match_index, 'match_score_cosine_similarity']/2)
        else:
            free_form_non_skill_question_overall_match_score_list.append(0.0)
    combined_match_score_record_free_form_non_skill_question = get_combined_match_score_record_free_form_non_skill_question_soft_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, free_form_non_skill_question_overall_match_score_list)
    return combined_match_score_record_free_form_non_skill_question

        
def get_combined_match_score_record_free_form_skill_question_soft_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_soft_skills, free_form_skill_question_overall_match_score_list):
    """
    
    Parameters
    ----------
    match_score_record_key_phrase_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Key_phrase extraction + Jaccard Similarity algorithm
        used for the skills question when mentee skill type is soft skill.
    match_score_record_cosine_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Doc2Vec + Cosine Similarity algorithm
        used for the skills question when mentee skill type is soft skill.
    match_score_record_classifier_soft_skills : pd.DataFrame
        Dataframe containing the match scores corresponding to the Soft Skill Classifier algorithm
        used for the skills question when mentee skill type is soft skill.
    free_form_skill_question_overall_match_score_list : pd.DataFrame
        List containing the overall match scores obtained from combining the match scores of all the  algorithms
        used for the skills question when mentee skill type is soft skill.

    Returns
    -------
    combined_match_score_record_free_form_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms 
        used for the skill question when mentee skill type is soft skill.

    """
    combined_match_score_record_free_form_skill_question = match_score_record_key_phrase_similarity.copy()
    combined_match_score_record_free_form_skill_question['match_score_cosine_similarity'] = match_score_record_cosine_similarity['match_score_cosine_similarity']
    combined_match_score_record_free_form_skill_question['match_score_soft_skill_matching'] = match_score_record_classifier_soft_skills['match_score_soft_skill_matching']
    combined_match_score_record_free_form_skill_question['match_score_hard_skill_matching'] = pd.Series(np.zeros(match_score_record_key_phrase_similarity.shape[0]))
    combined_match_score_record_free_form_skill_question['overall_match_score'] = pd.Series(free_form_skill_question_overall_match_score_list)
    return combined_match_score_record_free_form_skill_question

def get_combined_match_score_record_free_form_skill_question_hard_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_hard_skills, free_form_skill_question_overall_match_score_list):
    """
    
    Parameters
    ----------
    match_score_record_key_phrase_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Key_phrase extraction + Jaccard Similarity algorithm
        used for the skills question when mentee skill type is hard skill.
    match_score_record_cosine_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Doc2Vec + Cosine Similarity algorithm
        used for the skills question when mentee skill type is hard skill.
    match_score_record_classifier_hard_skills : pd.DataFrame
        Dataframe containing the match scores corresponding to the Hard Skill Classifier algorithm
        used for the skills question when mentee skill type is hard skill.
    free_form_skill_question_overall_match_score_list : pd.DataFrame
        List containing the overall match scores obtained from combining the match scores of all the  algorithms
        used for the skills question when mentee skill type is hard skill.

    Returns
    -------
    combined_match_score_record_free_form_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms 
        used for the skill question when mentee skill type is hard skill.

    """
    combined_match_score_record_free_form_skill_question = match_score_record_key_phrase_similarity.copy()
    combined_match_score_record_free_form_skill_question['match_score_cosine_similarity'] = match_score_record_cosine_similarity['match_score_cosine_similarity']
    combined_match_score_record_free_form_skill_question['match_score_soft_skill_matching'] = pd.Series(np.zeros(match_score_record_key_phrase_similarity.shape[0]))
    combined_match_score_record_free_form_skill_question['match_score_hard_skill_matching'] = match_score_record_classifier_hard_skills['match_score_hard_skill_matching']
    combined_match_score_record_free_form_skill_question['overall_match_score'] = pd.Series(free_form_skill_question_overall_match_score_list)
    return combined_match_score_record_free_form_skill_question

def get_combined_match_score_record_free_form_skill_question_soft_and_hard_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, match_score_record_classifier_soft_skills, match_score_record_classifier_hard_skills, free_form_skill_question_overall_match_score_list):
    """
    
    Parameters
    ----------
    match_score_record_key_phrase_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Key_phrase extraction + Jaccard Similarity algorithm
        used for the skills question when mentee skill type is soft and hard skill.
    match_score_record_cosine_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Doc2Vec + Cosine Similarity algorithm
        used for the skills question when mentee skill type is soft and hard skill.
    match_score_record_classifier_soft_skills : pd.DataFrame
        Dataframe containing the match scores corresponding to the Soft Skill Classifier algorithm
        used for the skills question when mentee skill type is soft and hard skill.
    match_score_record_classifier_hard_skills : pd.DataFrame
        Dataframe containing the match scores corresponding to the Hard Skill Classifier algorithm
        used for the skills question when mentee skill type is soft and hard skill.
    free_form_skill_question_overall_match_score_list : list
        List containing the overall match scores obtained from combining the match scores of all the  algorithms
        used for the skills question when mentee skill type is soft and hard skill.

    Returns
    -------
    combined_match_score_record_free_form_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms 
        used for the skill question when mentee skill type is soft and hard skill.

    """
    combined_match_score_record_free_form_skill_question = match_score_record_key_phrase_similarity.copy()
    combined_match_score_record_free_form_skill_question['match_score_cosine_similarity'] = match_score_record_cosine_similarity['match_score_cosine_similarity']
    combined_match_score_record_free_form_skill_question['match_score_soft_skill_matching'] = match_score_record_classifier_soft_skills['match_score_soft_skill_matching']
    combined_match_score_record_free_form_skill_question['match_score_hard_skill_matching'] = match_score_record_classifier_hard_skills['match_score_hard_skill_matching']
    combined_match_score_record_free_form_skill_question['overall_match_score'] = pd.Series(free_form_skill_question_overall_match_score_list)
    return combined_match_score_record_free_form_skill_question

def get_combined_match_score_record_free_form_non_skill_question_soft_skill_type(match_score_record_key_phrase_similarity, match_score_record_cosine_similarity, free_form_non_skill_question_overall_match_score_list):
    """
    
    Parameters
    ----------
    match_score_record_key_phrase_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Key_phrase extraction + Jaccard Similarity algorithm
        used for the non-skills question.
    match_score_record_cosine_similarity : pd.DataFrame
        Dataframe containing the match scores corresponding to the Doc2Vec + Cosine Similarity algorithm
        used for the non-skills question.
    free_form_non_skill_question_overall_match_score_list : list
        List containing the overall match scores obtained from combining the match scores of all the  algorithms
        used for the non-skills question.

    Returns
    -------
    combined_match_score_record_free_form_non_skill_question : pd.DataFrame
        Dataframe containing the match scores corresponding to all the freeform text matching algorithms 
        used for the non-skills question.

    """
    combined_match_score_record_free_form_non_skill_question = match_score_record_key_phrase_similarity.copy()
    combined_match_score_record_free_form_non_skill_question['match_score_cosine_similarity'] = match_score_record_cosine_similarity['match_score_cosine_similarity']
    combined_match_score_record_free_form_non_skill_question['overall_match_score'] = pd.Series(free_form_non_skill_question_overall_match_score_list)
    return combined_match_score_record_free_form_non_skill_question

#combined_free_form_match_score_record = get_combined_free_form_match_score_record(doc2vec_model, mentor_dataset.iloc[:10], mentee_dataset.iloc[:1], "soft_skills", mentor_dataset.iloc[:10], mentee_dataset.iloc[:1], mentor_dataset.iloc[:10], mentee_dataset.iloc[:1])

def objective_score( mentee_json, net_mentor_json):
    """
    
    Parameters
    ----------
    mentee_json : dict
        JSON containing mentee response extracted from the mentorship db.
    net_mentor_json : dict
        JSON containing all the mentor response extracted from the mentorship db.

    Returns
    -------
    results : list
        List containing the objective matching score 
        and the reason for match using objective matching for all the matches.

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path +"/disability_cluster.json",'r') as f:
        disability_dic = json.load(f)
    mentee_sector_flag = False
    mentee_disability_flag = False
    mentee_location_flag = False
    mentee_gender_flag = False
    mentee_isdisability = False
    mentee_sector = [mentee_json['industry'],mentee_json['subIndustry']]
    if 'SECTOR' in mentee_json['matchBasedOnMentee']:
        mentee_sector_flag = True
    if 'DISABILITY' in mentee_json['matchBasedOnMentee']:
        mentee_disability_flag = True
    if 'LOCATION' in mentee_json['matchBasedOnMentee']:
        mentee_location_flag = True
    if 'GENDER' in mentee_json['matchBasedOnMentee']:
        mentee_gender_flag = True
    if mentee_json['isPWD'] :
        mentee_isdisability = True
    mentee_disability_type = mentee_json['associatedDisabilities']
    mentee_gender = mentee_json['gender']
    mentee_location = [ mentee_json['location']['country'], mentee_json['location']['state'], mentee_json['location']['city']]
    mentee_experience = mentee_json['menteeWorkExp']
    results = []

    for mentor_json in net_mentor_json :
        mentor_sector_flag = False
        mentor_disability_flag = False
        mentor_location_flag = False
        mentor_gender_flag = False
        mentor_isdisability = False
        mentor_sector = [mentor_json['industry'],mentor_json['subIndustry']]
        experience = False
        if 'SECTOR' in mentor_json['matchBasedOnMentor']:
            mentor_sector_flag = True
        if 'DISABILITY' in mentor_json['matchBasedOnMentor']:
            mentor_disability_flag = True
        if 'LOCATION' in mentor_json['matchBasedOnMentor']:
            mentor_location_flag = True
        if 'GENDER' in mentor_json['matchBasedOnMentor']:
            mentor_gender_flag = True
        if 'EXPERIENCE' in mentor_json['matchBasedOnMentor']:
            experience = True
        if mentor_json['isPWD'] :
            mentor_disability_flag = True
        mentor_disability_type = mentor_json['associatedDisabilities']
        mentor_gender = mentor_json['gender']
        mentor_location = [ mentor_json['location']['country'], mentor_json['location']['state'], mentor_json['location']['city']]
        
        sector = mentee_sector_flag or mentor_sector_flag
        disability = mentor_disability_flag or mentee_disability_flag
        location = mentee_location_flag or mentor_location_flag
        gender = mentor_gender_flag or mentee_gender_flag

        count = 0
        net_score = 0
        reason_list = []

        # location score calculation
        score = 0
        count += 1
        if mentee_location[0] == mentor_location[0] :
            score += .5
            reason = "same country"
            if mentee_location[1] == mentor_location[1] :
                score += .35
                reason = "same state"
                if mentee_location[2] == mentor_location[2] :
                    score += .15
                    reason = "same city"
        if location : # increasing weightage for location score
            count += 1
            net_score += score
            if score >= .5 :
                reason_list.append(reason)
        net_score += score 

        # gender matching
        score = 0
        count += 1
        if mentee_gender == mentor_gender :
            score += 1
            reason = "same gender"
        if gender :  # increasing weightage for gender score
            count += 1
            net_score += score
            if score == 1 :
                reason_list.append(reason)
        net_score += score 

        # work experience matching
        score = 0
        count += 1
        if mentee_experience >= 1:
            score += .7
            if mentee_experience >= 3:
                score += .15
                if mentee_experience >= 5:
                    score += .15
        if experience : # increasing weightage for work_ex score
            count += 1
            net_score += score  #no reason here, as only mentor can select this option
        net_score += score 

        # sector matching
        score = 0
        count += 1
        if mentee_sector[0] == mentor_sector[0]:
            score += .6
            reason = "same sector"
            if mentee_sector[1] == mentor_sector[1]:
                score += .4
                reason = "same sector and same sub-sector"
        print(score)
        if sector : # increasing weightage
            count += 1
            net_score += score
            if score >= .6:
                reason_list.append(reason)
        net_score += score

        #disability matching
        score = 0
        count += 1
        total_disabilities = max(1, len( set(mentee_disability_type) | set(mentor_disability_type)) )# length of union of disablities between mentor and mentee
        if mentee_isdisability and mentor_isdisability :
            for mentee_disability in mentee_disability_type :
                temp_score = 0
                for mentor_disability in mentor_disability_type :  # finding best match for 'mentee_disability' in all of 'mentor_disability_type'
                    if mentor_disability in disability_dic[mentee_disability] :
                        score_temp = float (disability_dic[mentee_disability][mentor_disability])
                        if score_temp > temp_score : # selecting max possible score i.e. best match
                            temp_score = score_temp
                score += temp_score
        score = score / total_disabilities
        if disability : #more weightage
            count += 1
            net_score += score 
            if score > .5 :
                reason_list.append("similar disabilities")
        net_score += score 
        score = 0     
        net_score = net_score/count
        results.append ( [ net_score, "Objective Data : " + " ".join(reason_list) ] )

    return results


def generate_mentee_datasets_for_free_form_matching(mentee_instance):
    """

    Parameters
    ----------
    mentee_instance : dict
        JSON containing a single instance of response of the mentee extracted from the mentorship db.

    Returns
    -------
    mentee_dataset_skill_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the skills question.
    mentee_dataset_preference_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the preferences question.
    mentee_dataset_miscellaneous_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the miscellaneous question.

    """
    mentee_data_list_skill_question = []
    mentee_data_list_preference_question = []
    mentee_data_list_miscellaneous_question = []
    for mentee_index in range(1):
        mentee_id = str(mentee_instance['userId'])
        mentee_data_list_skill_question.append([mentee_id, mentee_instance['learnSkills']])
        mentee_data_list_preference_question.append([mentee_id, mentee_instance['menteesQualitiesPrefered']])
        mentee_data_list_miscellaneous_question.append([mentee_id, mentee_instance['anythingElseMentee']])
    mentee_dataset_skill_question = pd.DataFrame(mentee_data_list_skill_question, columns=['mentee_id', 'mentee_data'])
    mentee_dataset_preference_question = pd.DataFrame(mentee_data_list_preference_question, columns=['mentee_id', 'mentee_data'])
    mentee_dataset_miscellaneous_question = pd.DataFrame(mentee_data_list_miscellaneous_question, columns=['mentee_id', 'mentee_data'])
    return mentee_dataset_skill_question, mentee_dataset_preference_question, mentee_dataset_miscellaneous_question

def generate_mentor_datasets_for_free_form_matching(mentor_dic, mentee_instance):
    """
    
    Parameters
    ----------
    mentor_dic : dict
        JSON containing all the mentor response extracted from the mentorship db.
    mentee_instance : dict
        JSON containing a single instance of response of the mentee extracted from the mentorship db.

    Returns
    -------
    mentor_dataset_skill_question : pd.DataFrame
        Dataframe containing mentor ids and their text responses for the skills question.
    mentor_dataset_preference_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the preferences question.
    mentor_dataset_miscellaneous_question : pd.DataFrame
        Dataframe containing mentee ids and their text responses for the miscellaneous question.

    """
    mentor_dataframe = pd.DataFrame.from_records(list(mentor_dic.values()))
    mentor_data_list_skill_question = []
    mentor_data_list_preference_question = []
    mentor_data_list_miscellaneous_question = []
    for mentor_index in range(mentor_dataframe.shape[0]):
        mentor_id = str(mentor_dataframe.loc[mentor_index, 'userId'])
        if mentee_instance['kindOfSkillsMentee'] == mentor_dataframe.loc[mentor_index, 'kindOfSkillsMentor']:
            mentor_data_list_skill_question.append([mentor_id, mentor_dataframe.loc[mentor_index, 'mentorSkills']])
            mentor_data_list_preference_question.append([mentor_id, mentor_dataframe.loc[mentor_index, 'mentorQualitiesPrefered']])
            mentor_data_list_miscellaneous_question.append([mentor_id, mentor_dataframe.loc[mentor_index, 'anythingElseMentor']])
        else:
            mentor_data_list_skill_question.append([mentor_id, ""])
            mentor_data_list_preference_question.append([mentor_id, ""])
            mentor_data_list_miscellaneous_question.append([mentor_id, ""])
    mentor_dataset_skill_question = pd.DataFrame(mentor_data_list_skill_question, columns=['mentor_id', 'mentor_data'])
    mentor_dataset_preference_question = pd.DataFrame(mentor_data_list_preference_question, columns=['mentor_id', 'mentor_data'])
    mentor_dataset_miscellaneous_question = pd.DataFrame(mentor_data_list_miscellaneous_question, columns=['mentor_id', 'mentor_data'])
    return mentor_dataset_skill_question, mentor_dataset_preference_question, mentor_dataset_miscellaneous_question

def get_mentorship_recommendation_engine_final_match_record(combined_free_form_match_score_record, objective_score_dataframe):
    """
    
    Parameters
    ----------
    combined_free_form_match_score_record : pd.DataFrame
        Dataframe containing matches which are represented using the mentor and mentee ids
        and their corresponding match scores for all the matching algorithms used for freeform text.
        It also contains the overall match score for individual questions as well as the combined freeform text matching score.
        The reason for match is also present in the Dataframe.
    objective_score_dataframe : pd.DataFrame
        List containing the objective matching score 
        and the reason for match using objective matching for all the matches.

    Returns
    -------
    mentorship_recommendation_engine_final_match_record : pd.DataFrame
        Dataframe containing the final match score record and final reason for matching
        by combining the free form matching results and objective matching results.

    """
    mentorship_recommendation_engine_final_match_record = pd.concat([combined_free_form_match_score_record.loc[:, ['mentee_id', 'mentor_id', 'combined_free_form_match_score', 'free_form_reason_for_matching']], objective_score_dataframe], axis=1)
    mentorship_recommendation_engine_final_match_record['final_match_score'] = 0.6*mentorship_recommendation_engine_final_match_record['objective_score'] + 0.4*mentorship_recommendation_engine_final_match_record['combined_free_form_match_score']
    mentorship_recommendation_engine_final_match_record['final_reason_for_match'] = mentorship_recommendation_engine_final_match_record['objective_reason_for_matching'] + " " + mentorship_recommendation_engine_final_match_record['free_form_reason_for_matching']
    mentorship_recommendation_engine_final_match_record = mentorship_recommendation_engine_final_match_record.sort_values(by=['final_match_score'], ascending=False)
    return mentorship_recommendation_engine_final_match_record

def get_top_matches_mentorship(mentorship_recommendation_engine_final_match_record):
    """
    
    Parameters
    ----------
    mentorship_recommendation_engine_final_match_record : pd.DataFrame
        Dataframe containing the final match score record and final reason for matching
        by combining the free form matching results and objective matching results.

    Returns
    -------
    top_matches : list
        List of objects containing the mentor ids and reason for match for the top mentors 
        according to the final match score.

    """
    top_matches = []
    count_matches = min(3, mentorship_recommendation_engine_final_match_record.shape[0])
    for match_index in range(count_matches):
        match = {"mentor_id" : mentorship_recommendation_engine_final_match_record.loc[match_index, 'mentor_id'], "reason_for_match" : mentorship_recommendation_engine_final_match_record.loc[match_index, 'final_reason_for_match']}
        top_matches.append(match)
    return top_matches

