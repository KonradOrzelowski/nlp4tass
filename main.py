#%%
'''
get_insta_matrix  -> get connections using insta
get_sofifa_matrix -> get connections using sofia
plots graphs
calculate similarity
'''
import utils as utils
from tass_nlp import TassNlp
from tass_sofifa import TassSofia

# import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


# from datetime import datetime

# from keybert import KeyBERT
# from nltk.stem.wordnet import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer
# from keyphrase_vectorizers import KeyphraseCountVectorizer
#%% Insta

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
    # na = np.triu(na)
    
    
    df = pd.DataFrame(na) 
    df.columns = profiles
    df.index = profiles
    
    return df
      

idf = get_insta_matrix(profiles)

#%% Sofifa



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

# %% Insta graph

iG = nx.from_pandas_adjacency(idf)
pos = nx.spring_layout(iG)


_,weights = zip(*nx.get_edge_attributes(iG,'weight').items())

nodes = nx.draw_networkx_nodes(iG, pos, node_color="b")
edges = nx.draw_networkx_edges(iG, pos,
                               edge_color=weights, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(iG, pos, font_color="black")




fig = mpl.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('insta.png', dpi=360)
plt.show()
# %% Sofifa graph

sdf = get_sofifa_matrix(names, profiles)


sG = nx.from_pandas_adjacency(sdf)
pos = nx.spring_layout(sG)




_,weights = zip(*nx.get_edge_attributes(sG,'weight').items())

nodes = nx.draw_networkx_nodes(sG, pos, node_color="b")
edges = nx.draw_networkx_edges(sG, pos,
                                edge_color=weights, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(sG, pos, font_color="black")


fig = mpl.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('sofifa.png', dpi=360)
plt.show()

#%% similarity

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

print(f"jaccard_similarity {jaccard_similarity(iG.edges(), sG.edges())}")


GH = nx.compose(iG,sG)
GH.nodes()

# set edge colors
edge_colors = dict()
for edge in GH.edges():
    if iG.has_edge(*edge):
        if sG.has_edge(*edge):
            edge_colors[edge] = 'lightblue'
            continue
        edge_colors[edge] = 'green'
    elif sG.has_edge(*edge):
        edge_colors[edge] = 'red'

# set node colors
iG_nodes = set(iG.nodes())
sG_nodes = set(sG.nodes())
node_colors = []
for node in GH.nodes():
    if node in iG_nodes:
        if node in sG_nodes:
            node_colors.append('lightblue')
            continue
        node_colors.append('green')
    if node in sG_nodes:
        node_colors.append('red')
        
        
pos = nx.spring_layout(GH)

nodes = nx.draw_networkx_nodes(GH, pos, node_color="b")
edges = nx.draw_networkx_edges(GH, pos,
                                edgelist=edge_colors.keys(),
                                edge_color=edge_colors.values(),
                                edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(sG, pos, font_color="black")

fig = mpl.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('similarity.png', dpi=360)
plt.show()