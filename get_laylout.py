import utils as utils
from tass_nlp import TassNlp
from tass_sofifa import TassSofia

# import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


# from main import get_sofifa_matrix
profiles = ['harrykane', 'trentarnold66', 'sterling7', 'kylewalker2',
            'philfoden', 'reecejames', 'jackgrealish', 'ktrippier2',
            'declanrice', 'madders', 'bukayosaka87', 'masonmount',
            'andyrobertson94', 'johnstonesofficial', 'vardy7', 'aaronramsdale',
            'benchilwell', 'jpickford1', 'jhenderson', 'joegomez5',
            'kalvinphillips', 'marcusrashford', 'awbissaka']   

names = ['H. Kane', 'T. Alexander-Arnold', 'R. Sterling', 'K. Walker',
         'P. Foden', 'R. James', 'J. Grealish', 'K. Trippier',
         'D. Rice', 'J. Maddison', 'B. Saka', 'M. Mount',
         'A. Robertson', 'J. Stones', 'J. Vardy', 'A. Ramsdale',
         'B. Chilwell', 'J. Pickford', 'J. Henderson', 'J. Gomez',
         'K. Phillips', 'M. Rashford', 'A. Wan-Bissaka']


def get_sofifa_matrix(names, profiles):
    so = TassSofia('sofifa_processed.csv')
    
    footballers = {}
    
    for name in names:
      footballers[name] = so.get_footballer_atr(name).values.flatten()  


    df = []
    for i in names:
        lst = []
        for j in names:
            
            a = footballers[i]
            b = footballers[j]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            
            lst.append(cos_sim)
            
        df.append(lst)
    
    sa = np.array(df) 
    np.fill_diagonal(sa, 0)   
    row_sums = sa.sum(axis=1)
    sa = sa / row_sums[:, np.newaxis]
    # sa = np.triu(sa)
    
    df = pd.DataFrame(sa) 
    df.columns = profiles
    df.index = profiles
    
    return df


def get_sofifa_layout(names, profiles, laylout,\
                      file = 'pos_sofifa.csv') -> pd.DataFrame:

    sdf = get_sofifa_matrix(names, profiles)

    df = make_csv(sdf, names, profiles, laylout)
    df.to_csv(file, index=False)
    return df


def get_insta_matrix(profiles, threshold_kw = 0.5, threshold_cos = 0.5):


    tass_nlp = TassNlp()
    
    all_docs = tass_nlp.read_profiles(profiles)
    all_kw = tass_nlp.find_all_keywords(profiles, all_docs, threshold_kw)
    all_en = tass_nlp.encode_keywords(profiles, all_kw)

    df_list = []
    for i in profiles:
        lst = []
        for j in profiles:        
            similarity = utils.get_cos_similarity(all_en[i], all_en[j],
                                                  threshold = threshold_cos)
            lst.append(similarity)
            # print(f"{i} {j} similarity {similarity}")
        df_list.append(lst)
    
    na = np.array(df_list)
    np.fill_diagonal(na, 0)   
    row_sums = na.sum(axis=1)
    na = na / row_sums[:, np.newaxis]   
    
    df = pd.DataFrame(na) 
    df.columns = profiles
    df.index = profiles
    
    return df

def make_csv(sdf, names, profiles, laylout):
    sG = nx.from_pandas_adjacency(sdf)
    
    if laylout == 'spring_layout':
        pos = nx.spring_layout(sG)
    elif laylout == 'planar_layout':
        pos = nx.planar_layout(sG)
    elif laylout == 'kamada_kawai_layout':
        pos = nx.kamada_kawai_layout(sG)
    elif laylout == 'graphviz_layout':
        pos = nx.graphviz_layout(sG)
        
    df = pd.DataFrame(pos).T
    df.reset_index(inplace=True)
    df.columns=['player', 'x', 'y']
    
    return df
    
def get_insta_layout(names, profiles, laylout, \
                      file = 'pos_insta.csv'):

    sdf = get_insta_matrix(profiles)
    
    df = make_csv(sdf, names, profiles, laylout)
    df.to_csv(file, index=False)



def main():
    get_sofifa_layout(names, profiles, 'kamada_kawai_layout', 'pos_sofifa.csv')
    get_insta_layout(names, profiles, 'kamada_kawai_layout', 'pos_insta.csv')

# if __name__ == '__main__':
#     main()