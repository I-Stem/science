import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict


tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

all_stopwords = stopwords.words('english')

removed_stopwords = ['only', 'if', 'no', 'nor', 'not', 'other', 'than']
added_stopwords = ['mentor', 'mentee', 'mentors', 'mentees', 'mentorship',
                   'mentored', 'mentoring', 'guidance', 'guided', 'guide', 
                   'help', 'need', 'want', 'improve', 'get', 'better']
                   
for stopword in removed_stopwords:
    all_stopwords.remove(stopword)

for stopword in added_stopwords:
    all_stopwords.append(stopword)

all_stopwords = set(all_stopwords)


def get_nlp_preprocessed_dataset(users_dataset, users_type : str):
    """
    
    Parameters
    ----------
    users_dataset : pandas.DataFrame
        Dataset containing mentee or mentor text data.
    users_type : str
        Either mentee or mentor.

    Returns
    -------
    nlp_preprocessed_users_dataset : pandas.DataFrame
        Dataset after preprocessing of the text data of the mentee or mentor.

    """
    if users_type == "mentor":
        users_data_column_name = 'mentor_data'
    if users_type == "mentee":
        users_data_column_name = 'mentee_data'
    
    nlp_preprocessed_users_dataset = users_dataset.copy()
    for index in range(nlp_preprocessed_users_dataset.shape[0]):
        users_data = nlp_preprocessed_users_dataset.loc[index, users_data_column_name]
        users_data = re.sub("[^a-zA-Z]", " ", users_data)
        users_data = users_data.lower()
        tokens = word_tokenize(users_data)
        wordnet_lemmatizer = WordNetLemmatizer()
        corpus = []
        for token, tag in pos_tag(tokens):
            lemma = wordnet_lemmatizer.lemmatize(token, tag_map[tag[0]])
            if not token in all_stopwords:
                corpus.append(lemma)
        users_data = " ".join(corpus)
        nlp_preprocessed_users_dataset.loc[index, users_data_column_name] = users_data
    return nlp_preprocessed_users_dataset
