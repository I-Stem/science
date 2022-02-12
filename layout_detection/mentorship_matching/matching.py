import pandas as pd
import numpy as np
import os
from nltk import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from .mm_match_cosine_similarity.cosine_similarity import get_document_feature_vectors, get_cosine_similarity
from .mm_match_key_phrase_similarity.key_phrase_extraction import authenticate_ta_client, extract_key_phrases
from .mm_match_key_phrase_similarity.jaccard_similarity import get_jaccard_similarity_score
from .mm_match_skill_based_similarity.skill_similarity import train_skill_based_classifier_models, get_classification_models_dict
import joblib
from .mm_match_data_preprocessing.nlp_preprocessing import get_nlp_preprocessed_dataset

ta_client = authenticate_ta_client()

def get_match_based_on_soft_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset, nlp_preprocessed_mentee_dataset, classification_model_type = 'SVC_linear'):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_mentor_dataset : pd.DataFrame
        Dataframe containing mentor ids and their preprocessed text data for nlp.
    nlp_preprocessed_mentee_dataset : pd.DataFrame
        Dataframe containing mentee ids and their preprocessed text data for nlp.
    classification_model_type : str, optional
        Name of the classifier model type. The default is 'SVC_linear'.

    Returns
    -------
    match_score_record_soft_skill_classifier : pd.DataFrame
        Dataframe containing match score given by soft skill classification algorithm
        corresponding to match between mentee and mentor data.
        The matches are represented using the mentee_id and mentor_id.

    """
    if not os.path.exists('SVC_linear_mentee_soft_skill_classifier.sav'):
        mentee_dataset_skill_based_classifier_training = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentee_dataset.xlsx")
        nlp_preprocessed_mentee_dataset_skill_based_classifier_training = get_nlp_preprocessed_dataset(mentee_dataset_skill_based_classifier_training, "mentee")
        classification_models_dict = get_classification_models_dict()
        classifier_mentee_soft_skills, classification_mentee_soft_skills_models_dict = train_skill_based_classifier_models(doc2vec_model, nlp_preprocessed_mentee_dataset_skill_based_classifier_training, classification_models_dict, "mentee", "soft_skill_classifier")
    if not os.path.exists('SVC_linear_mentor_soft_skill_classifier.sav'):
        mentor_dataset_skill_based_classifier_training = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentor_dataset.xlsx")
        nlp_preprocessed_mentor_dataset_skill_based_classifier_training = get_nlp_preprocessed_dataset(mentor_dataset_skill_based_classifier_training, "mentor")
        classification_models_dict = get_classification_models_dict()
        classifier_mentor_soft_skills, classification_mentor_soft_skills_models_dict = train_skill_based_classifier_models(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_based_classifier_training, classification_models_dict, "mentor", "soft_skill_classifier")
    
    list_match_score_record_soft_skill_classifier = []
    for mentee_index in range(nlp_preprocessed_mentee_dataset.shape[0]):
        current_mentee_id = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_id']
        current_mentee_nlp_preprocessed_data = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_data']
        classification_mentee_soft_skill_models_dict = get_classification_models_dict()
        classification_mentee_soft_skill_models_dict[classification_model_type]['saved_file'] = classification_model_type + "_mentee_soft_skill_classifier.sav"
        for mentor_index in range(nlp_preprocessed_mentor_dataset.shape[0]):
            current_mentor_id = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_id']
            current_mentor_nlp_preprocessed_data = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_data']
            classification_mentor_soft_skill_models_dict = get_classification_models_dict()
            classification_mentor_soft_skill_models_dict[classification_model_type]['saved_file'] = classification_model_type + "_mentor_soft_skill_classifier.sav"
            match_score = get_match_score_skill_based_classifier(doc2vec_model, current_mentor_nlp_preprocessed_data, current_mentee_nlp_preprocessed_data, classification_mentor_soft_skill_models_dict, classification_mentee_soft_skill_models_dict, "soft_skill_classifier")
            list_match_score_record_soft_skill_classifier.append([current_mentee_id, current_mentor_id, match_score])
    match_score_record_soft_skill_classifier = pd.DataFrame(list_match_score_record_soft_skill_classifier, columns = ['mentee_id', 'mentor_id', 'match_score_soft_skill_matching'])
    min_max_scaler = MinMaxScaler()
    min_max_scaled_match_scores_soft_skill_matching = min_max_scaler.fit_transform(match_score_record_soft_skill_classifier['match_score_soft_skill_matching'].values.reshape(-1,1))
    match_score_record_soft_skill_classifier.drop('match_score_soft_skill_matching', axis=1, inplace=True)
    match_score_record_soft_skill_classifier['match_score_soft_skill_matching'] = pd.Series(min_max_scaled_match_scores_soft_skill_matching.reshape(-1,))
    return match_score_record_soft_skill_classifier

def get_match_based_on_hard_skill_classifier(doc2vec_model, nlp_preprocessed_mentor_dataset, nlp_preprocessed_mentee_dataset, classification_model_type = 'SVC_linear'):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_mentor_dataset : pd.DataFrame
        Dataframe containing mentor ids and their preprocessed text data for nlp.
    nlp_preprocessed_mentee_dataset : pd.DataFrame
        Dataframe containing mentee ids and their preprocessed text data for nlp.
    classification_model_type : str, optional
        Name of the classifier model type. The default is 'SVC_linear'.

    Returns
    -------
    match_score_record_hard_skill_classifier : pd.DataFrame
        Dataframe containing match score given by hard skill classification algorithm
        corresponding to match between mentee and mentor data.
        The matches are represented using the mentee_id and mentor_id.

    """
    if not os.path.exists('SVC_linear_mentee_hard_skill_classifier.sav'):
        mentee_dataset_skill_based_classifier_training = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentee_dataset.xlsx")
        nlp_preprocessed_mentee_dataset_skill_based_classifier_training = get_nlp_preprocessed_dataset(mentee_dataset_skill_based_classifier_training, "mentee")
        classification_models_dict = get_classification_models_dict()
        classifier_mentee_hard_skills, classification_mentee_hard_skills_models_dict = train_skill_based_classifier_models(doc2vec_model, nlp_preprocessed_mentee_dataset_skill_based_classifier_training, classification_models_dict, "mentee", "hard_skill_classifier")
        
    if not os.path.exists('SVC_linear_mentor_hard_skill_classifier.sav'):
        mentor_dataset_skill_based_classifier_training = pd.read_excel("mm_match_sample_dataset/question_1/sample_mentor_dataset.xlsx")
        nlp_preprocessed_mentor_dataset_skill_based_classifier_training = get_nlp_preprocessed_dataset(mentor_dataset_skill_based_classifier_training, "mentor")
        classification_models_dict = get_classification_models_dict()
        classifier_mentor_hard_skills, classification_mentor_hard_skills_models_dict = train_skill_based_classifier_models(doc2vec_model, nlp_preprocessed_mentor_dataset_skill_based_classifier_training, classification_models_dict, "mentor", "hard_skill_classifier")

    list_match_score_record_hard_skill_classifier = []
    for mentee_index in range(nlp_preprocessed_mentee_dataset.shape[0]):
        current_mentee_id = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_id']
        current_mentee_nlp_preprocessed_data = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_data']
        classification_mentee_hard_skill_models_dict = get_classification_models_dict()
        classification_mentee_hard_skill_models_dict[classification_model_type]['saved_file'] = classification_model_type + "_mentee_hard_skill_classifier.sav"
        for mentor_index in range(nlp_preprocessed_mentor_dataset.shape[0]):
            current_mentor_id = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_id']
            current_mentor_nlp_preprocessed_data = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_data']
            classification_mentor_hard_skill_models_dict = get_classification_models_dict()
            classification_mentor_hard_skill_models_dict[classification_model_type]['saved_file'] = classification_model_type + "_mentor_hard_skill_classifier.sav"
            match_score = get_match_score_skill_based_classifier(doc2vec_model, current_mentor_nlp_preprocessed_data, current_mentee_nlp_preprocessed_data, classification_mentor_hard_skill_models_dict, classification_mentee_hard_skill_models_dict, "hard_skill_classifier")
            list_match_score_record_hard_skill_classifier.append([current_mentee_id, current_mentor_id, match_score])
    match_score_record_hard_skill_classifier = pd.DataFrame(list_match_score_record_hard_skill_classifier, columns = ['mentee_id', 'mentor_id', 'match_score_hard_skill_matching'])
    min_max_scaler = MinMaxScaler()
    min_max_scaled_match_scores_hard_skill_matching = min_max_scaler.fit_transform(match_score_record_hard_skill_classifier['match_score_hard_skill_matching'].values.reshape(-1,1))
    match_score_record_hard_skill_classifier.drop('match_score_hard_skill_matching', axis=1, inplace=True)
    match_score_record_hard_skill_classifier['match_score_hard_skill_matching'] = pd.Series(min_max_scaled_match_scores_hard_skill_matching.reshape(-1,))
    return match_score_record_hard_skill_classifier
    
def get_match_score_skill_based_classifier(doc2vec_model, current_mentor_nlp_preprocessed_data : str, current_mentee_nlp_preprocessed_data : str, classification_mentor_skill_based_models_dict, classification_mentee_skill_based_models_dict, skill_classifier_type: str, classification_model_type = 'SVC_linear'):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    current_mentor_nlp_preprocessed_data : str
        Text data of mentor after preprocessing for nlp.
    current_mentee_nlp_preprocessed_data : str
        Text data of mentee after preprocessing for nlp.
    classification_mentor_skill_based_models_dict : dict
        It is the dictionary containing skill based classification models  for mentee data
        and their respective performance indicators and saved file names.
    classification_mentee_skill_based_models_dict : dict
        It is the dictionary containing skill based classification models  for mentor data
        and their respective performance indicators and saved file names.
    skill_classifier_type : str
        Either soft_skill_classifier or hard_skill_classifier.
    classification_model_type : TYPE, optional
        Name of the classifier model type. The default is 'SVC_linear'.

    Returns
    -------
    float
        Match score for similarity between mentee and mentor data
        based on probability of classification to similar class of soft skills
        given by the respective soft skill classfiers for mentee and mentor data.

    """
    if os.path.exists(classification_mentor_skill_based_models_dict[classification_model_type]['saved_file']):
        classifier_mentor_skill_based = joblib.load(classification_mentor_skill_based_models_dict[classification_model_type]['saved_file'])
    else:
        print(classification_model_type + "_mentor_" + skill_classifier_type + ".sav file not found. Use train_skill_based_classifier_models to train the model first.")
        return 0

    if os.path.exists(classification_mentee_skill_based_models_dict[classification_model_type]['saved_file']):
        classifier_mentee_skill_based = joblib.load(classification_mentee_skill_based_models_dict[classification_model_type]['saved_file'])
    else:
        print(classification_model_type + "_mentor_" + skill_classifier_type + ".sav file not found. Use train_skill_based_classifier_models to train the model first.")
        return 0
    
    feature_vector_mentor_data = get_document_feature_vectors(doc2vec_model, current_mentor_nlp_preprocessed_data)
    predicted_proba_vector_mentor = classifier_mentor_skill_based.predict_proba(feature_vector_mentor_data.reshape(-1, feature_vector_mentor_data.shape[0]))
    class_mentor_data = np.argmax(predicted_proba_vector_mentor, axis=1)[0]
    classification_score_mentor_data = predicted_proba_vector_mentor[0][class_mentor_data]
    
    feature_vector_mentee_data = get_document_feature_vectors(doc2vec_model, current_mentee_nlp_preprocessed_data)
    predicted_proba_vector_mentee = classifier_mentee_skill_based.predict_proba(feature_vector_mentee_data.reshape(-1, feature_vector_mentee_data.shape[0]))
    class_mentee_data = np.argmax(predicted_proba_vector_mentee, axis=1)[0]
    classification_score_mentee_data = predicted_proba_vector_mentee[0][class_mentee_data]

    if class_mentor_data == class_mentee_data:
        match_score = float(classification_score_mentor_data*classification_score_mentee_data)
        return match_score
    else:
        return 0
    
def get_match_based_on_cosine_similarity(doc2vec_model, nlp_preprocessed_mentor_dataset, nlp_preprocessed_mentee_dataset):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_mentor_dataset : pd.DataFrame
        Dataframe containing mentor ids and their preprocessed text data for nlp.
    nlp_preprocessed_mentee_dataset : pd.DataFrame
        Dataframe containing mentee ids and their preprocessed text data for nlp.

    Returns
    -------
    match_score_record_cosine_similarity : pd.DataFrame
        Dataframe containing match score given by doc2vec + Cosine Similarity algorithm
        corresponding to match between mentee and mentor data.
        The matches are represented using the mentee_id and mentor_id.

    """
    list_match_score_record_cosine_similarity = []
    for mentee_index in range(nlp_preprocessed_mentee_dataset.shape[0]):
        current_mentee_id = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_id']
        current_mentee_nlp_preprocessed_data = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_data']
        for mentor_index in range(nlp_preprocessed_mentor_dataset.shape[0]):
            current_mentor_id = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_id']
            current_mentor_nlp_preprocessed_data = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_data']
            match_score = get_match_score_cosine_similarity(doc2vec_model, current_mentor_nlp_preprocessed_data, current_mentee_nlp_preprocessed_data)
            list_match_score_record_cosine_similarity.append([current_mentee_id, current_mentor_id, match_score])
    match_score_record_cosine_similarity = pd.DataFrame(list_match_score_record_cosine_similarity, columns = ['mentee_id', 'mentor_id', 'match_score_cosine_similarity'])
    min_max_scaler = MinMaxScaler()
    min_max_scaled_match_scores_cosine_similarity = min_max_scaler.fit_transform(match_score_record_cosine_similarity['match_score_cosine_similarity'].values.reshape(-1,1))
    match_score_record_cosine_similarity.drop('match_score_cosine_similarity', axis=1, inplace=True)
    match_score_record_cosine_similarity['match_score_cosine_similarity'] = pd.Series(min_max_scaled_match_scores_cosine_similarity.reshape(-1,))
    return match_score_record_cosine_similarity

def get_match_score_cosine_similarity(doc2vec_model, current_mentor_nlp_preprocessed_data : str, current_mentee_nlp_preprocessed_data : str):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    current_mentor_nlp_preprocessed_data : str
        Text data of mentor after preprocessing for nlp.
    current_mentee_nlp_preprocessed_data : str
        Text data of mentee after preprocessing for nlp.

    Returns
    -------
    match_score_cosine_similarity : float
        Match score for similarity between mentee and mentor data
        based on cosine similarity of feature vectors 
        given by the doc2vec_model for mentee and mentor data.

    """
    try:
        current_mentor_data_feature_vector = get_document_feature_vectors(doc2vec_model, current_mentor_nlp_preprocessed_data)
        current_mentee_data_feature_vector = get_document_feature_vectors(doc2vec_model, current_mentee_nlp_preprocessed_data)
        match_score_cosine_similarity = get_cosine_similarity(current_mentor_data_feature_vector, current_mentee_data_feature_vector)
        if match_score_cosine_similarity < 0.0:
            match_score_cosine_similarity = 0.0
    except:
        match_score_cosine_similarity = 0.0
    return match_score_cosine_similarity

def get_match_based_on_key_phrase_similarity(nlp_preprocessed_mentor_dataset, nlp_preprocessed_mentee_dataset):
    """
    
    Parameters
    ----------
    nlp_preprocessed_mentor_dataset : pd.DataFrame
        Dataframe containing mentor ids and their preprocessed text data for nlp.
    nlp_preprocessed_mentee_dataset : pd.DataFrame
        Dataframe containing mentee ids and their preprocessed text data for nlp.

    Returns
    -------
    match_score_record_key_phrase_similarity : pd.DataFrame
        Dataframe containing match score given by Keyphrase Extraction + Jaccard Similarity algorithm
        corresponding to match between mentee and mentor data.
        The matches are represented using the mentee_id and mentor_id.

    """
    list_match_score_record_key_phrase_similarity = []
    for mentee_index in range(nlp_preprocessed_mentee_dataset.shape[0]):
        current_mentee_id = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_id']
        current_mentee_nlp_preprocessed_data = nlp_preprocessed_mentee_dataset.loc[mentee_index, 'mentee_data']
        for mentor_index in range(nlp_preprocessed_mentor_dataset.shape[0]):
            current_mentor_id = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_id']
            current_mentor_nlp_preprocessed_data = nlp_preprocessed_mentor_dataset.loc[mentor_index, 'mentor_data']
            match_score = get_match_score_key_phrase_similarity(current_mentor_nlp_preprocessed_data, current_mentee_nlp_preprocessed_data)
            list_match_score_record_key_phrase_similarity.append([current_mentee_id, current_mentor_id, match_score])
    match_score_record_key_phrase_similarity = pd.DataFrame(list_match_score_record_key_phrase_similarity, columns = ['mentee_id', 'mentor_id', 'match_score_key_phrase_similarity'])
    min_max_scaler = MinMaxScaler()
    min_max_scaled_match_scores_key_phrase_similarity = min_max_scaler.fit_transform(match_score_record_key_phrase_similarity['match_score_key_phrase_similarity'].values.reshape(-1,1))
    match_score_record_key_phrase_similarity.drop('match_score_key_phrase_similarity', axis=1, inplace=True)
    match_score_record_key_phrase_similarity['match_score_key_phrase_similarity'] = pd.Series(min_max_scaled_match_scores_key_phrase_similarity.reshape(-1,))
    return match_score_record_key_phrase_similarity

def get_match_score_key_phrase_similarity(current_mentor_nlp_preprocessed_data : str, current_mentee_nlp_preprocessed_data : str):
    """
    
    Parameters
    ----------
    current_mentor_nlp_preprocessed_data : str
        Text data of mentor after preprocessing for nlp.
    current_mentee_nlp_preprocessed_data : str
        Text data of mentee after preprocessing for nlp.

    Returns
    -------
    match_score_key_phrase_similarity : float
        Match score for similarity between mentee and mentor data
        based on Jaccard similarity of lists containing extracted key phrases 
        given by the Key Phrase Extraction API for mentee and mentor data.

    """
    current_mentor_data_list = [current_mentor_nlp_preprocessed_data]
    current_mentee_data_list = [current_mentee_nlp_preprocessed_data]
    current_mentor_key_phrases = extract_key_phrases(ta_client, current_mentor_data_list)
    current_mentor_tokenized_key_phrases = [word_tokenize(key_phrase) for key_phrase in current_mentor_key_phrases]
    current_mentee_key_phrases = extract_key_phrases(ta_client, current_mentee_data_list)
    current_mentee_tokenized_key_phrases = [word_tokenize(key_phrase) for key_phrase in current_mentee_key_phrases]
    match_score = 0
    for current_mentor_tokenized_key_phrase in current_mentor_tokenized_key_phrases:
        for current_mentee_tokenized_key_phrase in current_mentee_tokenized_key_phrases:
            match_score += get_jaccard_similarity_score(current_mentor_tokenized_key_phrase, current_mentee_tokenized_key_phrase)
    try:
        match_score_key_phrase_similarity = float(match_score) / float((len(current_mentor_tokenized_key_phrases)+len(current_mentee_tokenized_key_phrases))/2)
        return match_score_key_phrase_similarity
    except:
        return 0

