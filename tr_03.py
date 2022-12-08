# %%
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

import torch

from transformers import AutoTokenizer, AutoModel
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
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

# %% read all docs
from nltk.stem.wordnet import WordNetLemmatizer

# read all docs without emoji
def read_profiles(profiles, lemmatizer):
    all_docs = {}
    for profile in profiles:
        all_docs[profile] = utils.read_all_lines_no_emoji(lemmatizer, profile)
    return all_docs


profiles = ['garethbale11', 'harrykane']
lemmatizer = WordNetLemmatizer()

all_docs = read_profiles(profiles, lemmatizer)

all_docs['garethbale11'][0]
# %% extract keywords

def find_all_keywords(profiles, all_docs, kw_model, vectorizer) -> dict:
    all_kw = {}
    for profile in profiles:
        all_kw[profile] = utils.extract_kw_from_docs(
            all_docs[profile], kw_model, vectorizer)
        all_kw[profile] = utils.remove_sim_kw(all_kw[profile])
    return all_kw


kw_model = KeyBERT()
vectorizer = KeyphraseCountVectorizer()
key_words = find_all_keywords(profiles, all_docs, kw_model, vectorizer)

key_words['garethbale11'][0]
# %% encode keywords
def encode_keywords(profiles: list[str], key_words: dict, model: SentenceTransformer) -> dict:
    encoding = {}
    for profile in profiles:
        encoding[profile] = model.encode(key_words[profile], show_progress_bar=True)
    return encoding

# Load AutoModel from huggingface model repository
model = SentenceTransformer('all-MiniLM-L6-v2')
all_encoding = encode_keywords(profiles, key_words, model)

all_encoding['garethbale11'][0]
# %%
