import numpy as np
import os
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def create_tagged_document(list_of_list_of_words):
    """
    
    Parameters
    ----------
    list_of_list_of_words : list
        List of list of words used to generate the tagged document.

    Yields
    ------
    gensim.models.doc2vec.TaggedDocument
        Tagged document to be used for training the doc2vec_model.

    """
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield TaggedDocument(list_of_words, [i])

def get_doc2vec_model():
    """
    
    Returns
    -------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.

    """
    if not os.path.exists("doc2vec_model.bin"):
        # Download dataset
        dataset = api.load("text8")
        data = [d for d in dataset]
        
        train_data = list(create_tagged_document(data))
        # Init the Doc2Vec model
        model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        # Build the Volabulary
        model.build_vocab(train_data)
        # Train the Doc2Vec model
        model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("doc2vec_model.bin")
    doc2vec_model = Doc2Vec.load("doc2vec_model.bin")
    return doc2vec_model

def get_document_feature_vectors(doc2vec_model, document: str):
    """
    
    Parameters
    ----------
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The doc2vec_model to convert text data to vectors.
    document : str
        The document that is to be converted to feature vector.

    Returns
    -------
    np.array
        Feature vector for the document.

    """
    return doc2vec_model.infer_vector(word_tokenize(document))

def get_cosine_similarity(feature_vec_1, feature_vec_2):
    """
    
    Parameters
    ----------
    feature_vec_1 : np.array
        Feature vector for 1st document.
    feature_vec_2 : np.array
        Feature vector for 2nd document.

    Returns
    -------
    float
        Cosine similarity score between the two feature vectors.

    """    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

