import glob
import re
import functools
from sentence_transformers import util
from datetime import datetime

# TassNlp: read_profiles
def remove_emojis(string: str) -> str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji_pattern, '', string)

# TassNlp: read_profiles
def read_all_lines_no_emoji(lemmatizer, profile_name: str) -> list:
    
    txt_files = glob.glob(f"{profile_name}/*.txt")
    docs = []
    for txt in txt_files:
        with open(txt, 'r' , encoding="utf-8") as fd:
            all_of_it = fd.read()
            
        docs.append(remove_emojis(str(lemmatizer.lemmatize(all_of_it))))
    return docs


# TassNlp: read_profiles
def read_from_time_range(lemmatizer, profile_name: str, start_date, end_date) -> list:
    
    txt_files = glob.glob(f"{profile_name}/*.txt")
    docs = []
    
    for txt in txt_files:
        txt_data = txt.split("\\")[1].replace('_UTC.txt', '')
        txt_data = datetime.strptime(txt_data, '%Y-%m-%d_%H-%M-%S')
        if (txt_data > start_date) and (txt_data < end_date):
            with open(txt, 'r' , encoding="utf-8") as fd:
                all_of_it = fd.read()
            docs.append(remove_emojis(str(lemmatizer.lemmatize(all_of_it))))
    
    return(docs)


# TassNlp: find_all_keywords
def extract_kw_from_docs(docs, model, vec):
    key_words = model.extract_keywords(docs=docs, 
                                          vectorizer=vec)
    lst = []
    for i in key_words:
        lst.append(list((dict(i).keys())))
        
    lst = functools.reduce(lambda x,y: x+y,lst)
    
    return lst

    
# TassNlp: find_all_keywords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def remove_sim_kw(documents: list[str], threshold:float) -> list[str]:
    '''
    Remove key words with similar meaning

    Parameters
    ----------
    documents : list[str]
        key words.

    Returns
    -------
    list[str]
        Unique key words.

    '''
    documents = list(set(documents))
    
    tfidf = TfidfVectorizer().fit_transform(documents)
    # no need to normalize, since Vectorizer will return normalized tf-idf
    pairwise_similarity = tfidf * tfidf.T
    
    pairwise_similarity = pairwise_similarity.toarray()
    np.fill_diagonal(pairwise_similarity,0)
    
    
    pairwise_similarity[np.argmax(pairwise_similarity, axis=1)]
    
    
    max_idx = np.argmax(pairwise_similarity, axis=1)
    
    
    pairwise_similarity
    
    for idx, row in enumerate(pairwise_similarity):
        if(pairwise_similarity[idx][max_idx[idx]] >= threshold):
            documents[idx]= documents[max_idx[idx]]
    return list(set(documents))


def get_cos_similarity(nd1: np.ndarray, nd2: np.ndarray,
                                             threshold = 0.5) -> np.float32:
    
    cos_ndarray = util.pytorch_cos_sim(nd1, nd2).numpy()
    
    return np.sum(cos_ndarray[cos_ndarray>threshold])