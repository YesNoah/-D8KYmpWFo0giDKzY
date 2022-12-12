import pandas as pd
from re import sub
from gensim.utils import simple_preprocess

def data_get(path_to_data = "data\\raw\\potential-talents - Aspiring human resources - seeking human resources.csv"):
    df = pd.read_csv(path_to_data)
    return df

def data_getridofstrings(df):
    df['connection'] = pd.to_numeric(df['connection'],errors='coerce')
    df['connection'] = df['connection'].fillna(500)
    return df

def preprocess(doc):
    stpwrds = ['the', 'and', 'are', 'a', 'for', 'at', 'of', 'is', 'an']
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stpwrds]
