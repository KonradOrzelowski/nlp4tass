#%%
from datetime import datetime

from tass_instaloader import TassInstaloader
from other.info_insta import insta_dict

# from utils import read_all_lines_no_emoji, extract_kw_from_docs, heatmap
import utils as utils
# profile_name = 'harrykane'
# tL = TassInstaloader(insta_dict['user'], insta_dict['password'])

# profile = tL.get_profile(profile_name)
# posts = profile.get_posts()

# singe = datetime(2022, 9, 1)
# until = datetime(2022, 11, 30)

# tL.download_post_from_time_range(profile, posts, singe, until)

#%% read all docs
from nltk.stem.porter import PorterStemmer


from nltk.stem.wordnet import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

all_docs = {}
all_docs['garethbale11'] = utils.read_all_lines_no_emoji(lemmatizer, 'garethbale11')
all_docs['harrykane'] = utils.read_all_lines_no_emoji(lemmatizer, 'harrykane')

#%% docs processing
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.wordnet import WordNetLemmatizer

# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# print("Original Word: 'studies' ")
# print()
# print('With Stemming: ' + str(stemmer.stem("studies")))
# print('with Lemmatization: ' + str(lemmatizer.lemmatize("studies")))


# for doc in all_docs['garethbale11']:
#     print(doc)



#%% keywords

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

key_words = {}

vectorizer = KeyphraseCountVectorizer()
kw_model = KeyBERT()

key_words['garethbale11'] = utils.extract_kw_from_docs(all_docs['garethbale11'], kw_model, vectorizer)
key_words['harrykane'] = utils.extract_kw_from_docs(all_docs['harrykane'], kw_model, vectorizer)

key_words['garethbale11'] = utils.remove_sim_kw(key_words['garethbale11'])
key_words['harrykane'] = utils.remove_sim_kw(key_words['harrykane'])

#%% 
# lemmatizer = WordNetLemmatizer()
# kks = [stemmer.stem(kdoc) for kdoc in key_words['harrykane']]
# kkl = [lemmatizer.lemmatize(kdoc) for kdoc in key_words['harrykane']]

# words = word_tokenize(key_words['garethbale11'])
# text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in words])
#%%
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import util

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_sentence_embeddings(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings



# gb = get_sentence_embeddings(key_words['garethbale11'])
# hk = get_sentence_embeddings(key_words['harrykane'])

# res = util.pytorch_cos_sim(gb, hk)
# util.pytorch_cos_sim(gb, hk)

# import torch.nn.functional as F

# cosine similarity = normalize the vectors & multiply
# C = F.normalize(gb) @ F.normalize(hk).t()


# cluster.util.cosine_distance(F.normalize(gb) , F.normalize(hk).t())


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sentences we want to encode. Example:
# sentence = ['This framework generates embeddings for each input sentence']


#Sentences are encoded by calling model.encode()
egb = model.encode(key_words['garethbale11'], show_progress_bar  = True)
ehk = model.encode(key_words['harrykane'], show_progress_bar  = True)


# tf.reduce_mean(all_embeddings, axis=1)