import glob
import re
import functools

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


# from nltk.stem.wordnet import WordNetLemmatizer

# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# print("Original Word: 'studies' ")
# print()
# print('With Stemming: ' + str(stemmer.stem("studies")))
# print('with Lemmatization: ' + str(lemmatizer.lemmatize("studies")))

def read_all_lines_no_emoji(stemmer, profile_name: str) -> list:
    
    txt_files = glob.glob(f"{profile_name}/*.txt")
    docs = []
    for txt in txt_files:
        with open(txt, 'r' , encoding="utf-8") as fd:
            all_of_it = fd.read()
            
        docs.append(remove_emojis(str(stemmer.lemmatize(all_of_it))))
    return docs

def extract_kw_from_docs(docs, model, vec):
    key_words = model.extract_keywords(docs=docs, 
                                          vectorizer=vec)
    lst = []
    for i in key_words:
        lst.append(list((dict(i).keys())))
        
    lst = functools.reduce(lambda x,y: x+y,lst)
    
    return lst




import matplotlib.pyplot as plt
import numpy as np
def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f"%values[i, j],
                           ha="center", va="center", color="w", fontsize=6)

    fig.tight_layout()
    plt.show()
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def remove_sim_kw(documents: list[str]) -> list[str]:
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
        if(pairwise_similarity[idx][max_idx[idx]] >= 0.5):
            documents[idx]= documents[max_idx[idx]]
    return list(set(documents))



def call_sim(p2_docs1: list[str], p2_docs: list[str]) -> float:

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
        if(pairwise_similarity[idx][max_idx[idx]] >= 0.5):
            documents[idx]= documents[max_idx[idx]]
    return list(set(documents))


## find all keywrods from all profiles
def find_all_kw(profiles: list[str], model, stemmer, vec) -> list[str]:
    """
    
    """
    all_kw = []
    for profile in profiles:
        docs = read_all_lines_no_emoji(stemmer, profile)

        all_kw.extend(remove_sim_kw(extract_kw_from_docs(docs, model, vec)))
    return all_kw