#%%
import utils as utils

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


from datetime import datetime

from keybert import KeyBERT
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from keyphrase_vectorizers import KeyphraseCountVectorizer
#%%

class TassNlp:
    def __init__(self):

        self.lemmatizer = WordNetLemmatizer()
        self.kw_model = KeyBERT()
        self.vectorizer = KeyphraseCountVectorizer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def read_profiles(self, profiles: list[str], start_date = datetime(2000, 11, 1),
                                                 end_date = datetime(2023, 1, 5)) -> dict:
        '''Read all docs without emoji'''
        all_docs = {}
        for profile in profiles:
            all_docs[profile] = utils.read_from_time_range(self.lemmatizer, profile,
                                                           start_date, end_date)
        return all_docs

    def find_all_keywords(self, profiles: list[str], all_docs: dict, threshold:float) -> dict:
        all_kw = {}
        for profile in profiles:
            all_kw[profile] = utils.extract_kw_from_docs(
                all_docs[profile], self.kw_model, self.vectorizer)
            all_kw[profile] = utils.remove_sim_kw(all_kw[profile], threshold)
        return all_kw

    def encode_keywords(self, profiles: list[str], key_words: dict) -> dict:
        encoding = {}
        for profile in profiles:
            encoding[profile] = self.sentence_transformer.encode(key_words[profile],
                                                                 show_progress_bar=True)
        return encoding
    

# %%
def main():
    '''
    read docs -> get keywords -> encode keywords -> get cos
    
    Returns
    -------
    None.
    
    '''
    profiles = ['harrykane', 'trentarnold66', 'sterling7', 'kylewalker2',
                'philfoden', 'reecejames', 'jackgrealish', 'ktrippier2',
                'declanrice', 'madders', 'bukayosaka87', 'masonmount']


    tass_nlp = TassNlp()
    
    all_docs = tass_nlp.read_profiles(profiles)
    all_kw = tass_nlp.find_all_keywords(profiles, all_docs, threshold=0.5)
    all_en = tass_nlp.encode_keywords(profiles, all_kw)
    
if __name__ == '__main__':
    main()
    
