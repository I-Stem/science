import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib
from ..mm_match_cosine_similarity.cosine_similarity import get_document_feature_vectors

def get_classification_models_dict():
    """
    
    Returns
    -------
    classification_models_dict : dict
        It is the dictionary containing classification models 
        and their respective performance indicators and saved file names.

    """
    classification_models_dict = {'LR' : {}, 'KNN' : {}, 'SVC_linear' : {}, 'SVC_rbf' : {}, 'GaussianNB' : {}, 'DTC' : {}, 'RFC' : {}}
        
    for classification_model_type in classification_models_dict:
        classification_models_dict[classification_model_type]['saved_file'] = classification_model_type + '.sav'
        classification_models_dict[classification_model_type]['model'] = None
        classification_models_dict[classification_model_type]['conf_matrix'] = None
        classification_models_dict[classification_model_type]['accuracy'] = None
        classification_models_dict[classification_model_type]['accuracy_train'] = None
        classification_models_dict[classification_model_type]['precision'] = None
        classification_models_dict[classification_model_type]['recall'] = None
    
    classification_models_dict['LR']['model'] = LogisticRegression(random_state=0, max_iter=10000)
    classification_models_dict['KNN']['model'] = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classification_models_dict['SVC_linear']['model'] = SVC(kernel = 'linear', random_state = 0, max_iter=10000, probability=True)
    classification_models_dict['SVC_rbf']['model'] = SVC(kernel = 'rbf', max_iter=10000, random_state = 0)
    classification_models_dict['GaussianNB']['model'] = GaussianNB()
    classification_models_dict['DTC']['model'] = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classification_models_dict['RFC']['model'] = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
    
    return classification_models_dict

def convert_dataset_for_training_classifier_models(doc2vec_model, dataset, user_type: str, model_type: str):
    """
    
    Parameters
    ----------
    doc2vec_model: gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    dataset : pd.DataFrame
        The dataset to be converted for classifier model use.
    user_type : str
        Either mentee or mentor.
    model_type : str
        Either soft_skill_classifier or hard_skill_classifier.

    Returns
    -------
    converted_features, converted_target : np.array, np.array
        Features and target for use of the classifier models.
    
    """
    try:
        converted_data_vector_list = []
        converted_target_vector_list = []
        if model_type == "soft_skill_classifier":
            soft_skill_vector = dataset.loc[:, ['general', 'disability', 'interview', 'abroad_education']].values
            soft_skill_vector_to_decimal_list = []
            for index in range(soft_skill_vector.shape[0]):
                soft_skill_vector_to_decimal_list.append(int("".join(str(x) for x in soft_skill_vector[index]), 2))
            le = LabelEncoder()
            soft_skill_target_labels = le.fit_transform(np.array(soft_skill_vector_to_decimal_list))
            ohe = OneHotEncoder()
            ohe_encoded_soft_skill_target = ohe.fit_transform(soft_skill_target_labels.reshape(-1,1)).toarray()
        
        if model_type == "hard_skill_classifier":
            le = LabelEncoder()
            hard_skill_target_labels = le.fit_transform(dataset.loc[:, 'hard_skill_type'])
            ohe = OneHotEncoder()
            ohe_encoded_hard_skill_target = ohe.fit_transform(hard_skill_target_labels.reshape(-1,1)).toarray()
            
        for index in range(dataset.shape[0]):
            data = dataset.loc[index, user_type + "_data"]
            data_vector = get_document_feature_vectors(doc2vec_model, data)
            if model_type == "soft_skill_classifier":
                target_vector = ohe_encoded_soft_skill_target[index, :]
            elif model_type == "hard_skill_classifier":
                target_vector = ohe_encoded_hard_skill_target[index, :]
            converted_data_vector_list.append(data_vector)
            converted_target_vector_list.append(target_vector)
        
        converted_features = np.array(converted_data_vector_list)
        converted_target = np.array(converted_target_vector_list)
        return converted_features, converted_target
    except:
        print("Incorrect user_type or model_type")
        return None, None

def undersample_majority_class(features, target, majority_class: int, reduced_size_of_majority: int):
    """
    
    Parameters
    ----------
    features : np.array
        Input for the classifier models.
    target : np.array
        Output for the classifier models.
    majority_class : int
        Index in the Target which corresponds to the majority class.
    reduced_size_of_majority : int
        Size of majority class after undersampling.

    Returns
    -------
    undersampled_features, undersampled_target: np.array, np.array
        Features and target after undersampling.

    """
    majority_indices_list = []
    undersampled_features_list = []
    undersampled_target_list = []
    for index in range(target.shape[0]):
        if target[index, majority_class] == 1:
            majority_indices_list.append(index)
        else:
            undersampled_features_list.append(features[index, :])
            undersampled_target_list.append(target[index, :])
    undersampled_majority_size = 0
    while(undersampled_majority_size < reduced_size_of_majority):
        majority_index_to_add = majority_indices_list[random.randint(0, len(majority_indices_list)-1)]
        undersampled_features_list.append(features[majority_index_to_add, :])
        undersampled_target_list.append(target[majority_index_to_add, :])
        undersampled_majority_size += 1
    
    undersampled_features = np.array(undersampled_features_list)
    undersampled_target = np.array(undersampled_target_list)
    return undersampled_features, undersampled_target

def convert_target_to_labels(target):
    """

    Parameters
    ----------
    target : np.array
        Multidiminesional target matrix.

    Returns
    -------
    target_labels : np.array
        Target Label Vector obtained from target matrix.

    """
    target_labels_list = []
    for index in range(target.shape[0]):
        target_labels_list.append(np.where(target[index]==1)[0][0])
    target_labels = np.array(target_labels_list)
    return target_labels
    
def train_classifier_model(classifier_user, classification_user_models_dict, classification_model_type, features_train, target_train, features_test, target_test):
    """
    
    Parameters
    ----------
    classifier_user : model
        Classifier model to be used when no saved model found on disk.
    classification_user_models_dict : dict
        It is the dictionary containing classification models 
        and their respective performance indicators and saved file names.
    classification_model_type : str
        Name of the classifier model type.
    features_train : np.array
        Features matrix to be used for training the classifier model.
    target_train : np.array
        Target Label Vector to be used for training the classifier model.
    features_test : np.array
        Features matrix to be used for testing the classifier model.
    target_test : np.array
        Target Label Vector to be used for testing the classifier model.

    Returns
    -------
    classifier : model
        Classifier model either after training or from the saved file.
    classification_user_models_dict : dict
        Updated classification_user_models_dict dictionary including the trained classifier model
        along with its performance indicators and saved file name.

    """
    if os.path.exists(classification_user_models_dict[classification_model_type]['saved_file']):
        classifier = joblib.load(classification_user_models_dict[classification_model_type]['saved_file'])
    else:
        classifier = classifier_user
        classifier.fit(features_train, target_train)
        # save the classifier model to disk
        filename = classification_user_models_dict[classification_model_type]['saved_file']
        joblib.dump(classifier, filename)
    
    classification_user_models_dict[classification_model_type]['model'] = classifier
    
    # Predicting the Test set results
    target_pred_proba = classifier.predict_proba(features_test)
    target_pred = np.argmax(target_pred_proba, axis=1)
    
    #Predicting the Train set results
    target_train_pred_proba = classifier.predict_proba(features_train)
    target_train_pred = np.argmax(target_train_pred_proba, axis=1)
    
    # Getting the metrics and confusion matrix for results
    cm = confusion_matrix(target_test, target_pred)
    print(cm)    
    acc_train = accuracy_score(target_train, target_train_pred)
    print(acc_train)    
    acc = accuracy_score(target_test, target_pred)
    print(acc)
    precision = precision_score(target_test, target_pred, average='weighted')
    print(precision)
    recall = recall_score(target_test, target_pred, average='weighted')
    print(recall)
    
    classification_user_models_dict[classification_model_type]['conf_matrix'] = cm
    classification_user_models_dict[classification_model_type]['accuracy_train'] = acc_train
    classification_user_models_dict[classification_model_type]['accuracy'] = acc
    classification_user_models_dict[classification_model_type]['precision'] = precision
    classification_user_models_dict[classification_model_type]['recall'] = recall
    
    return classifier, classification_user_models_dict

def train_skill_based_classifier_models(doc2vec_model, nlp_preprocessed_user_dataset, classification_models_dict, user_type: str, skill_classifier_type: str, classification_model_type = 'SVC_linear'):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    nlp_preprocessed_user_dataset : np.array
        The dataset to be used for training and testing of classifier models.
    classification_models_dict : dict
        It is the dictionary containing classification models 
        and their respective performance indicators and saved file names.
    user_type : str
        Either mentee or mentor.
    skill_classifier_type : str
        Either soft_skill_classifier or hard_skill_classifier.
    classification_model_type : str, optional
        Name of the classifier model type. The default is 'SVC_linear'.

    Returns
    -------
    classifier_user_trained : model
        Classifier model either after training or from the saved file.
    classification_user_updated_models_dict : dict
        Updated classification_user_models_dict dictionary including the trained classifier model
        along with its performance indicators and saved file name.

    """
    features, target = convert_dataset_for_training_classifier_models(doc2vec_model, nlp_preprocessed_user_dataset, user_type, skill_classifier_type)
    count_of_samples_of_classes_in_target = np.sum(target, axis=0)
    count_of_samples_of_minority_class_in_target = count_of_samples_of_classes_in_target.min(axis=0)
    for class_index in range(target.shape[1]):
        if count_of_samples_of_classes_in_target[class_index] > count_of_samples_of_minority_class_in_target:
            features, target = undersample_majority_class(features, target, class_index, count_of_samples_of_minority_class_in_target)
    target = convert_target_to_labels(target)
       
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    features_train = sc.fit_transform(features_train)
    features_test = sc.transform(features_test)
        
    classification_user_models_dict = classification_models_dict.copy()
    classification_user_models_dict[classification_model_type]['saved_file'] = classification_model_type + "_" + user_type + "_" + skill_classifier_type + ".sav"
    classifier_user = classification_user_models_dict[classification_model_type]['model']
    
    classifier_user_trained, classification_user_updated_models_dict = train_classifier_model(classifier_user, classification_user_models_dict, classification_model_type, features_train, target_train, features_test, target_test)
        
    return classifier_user_trained, classification_user_updated_models_dict
