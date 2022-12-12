import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from re import sub
from gensim.utils import simple_preprocess
import rake_nltk
from rake_nltk import Rake
import data.data_utils as data_utils


nltk.download('wordnet')

def main_func(path_to_data):
    
    #Data_Get
    df = data_utils.data_get(path_to_data)

    #get rid of pesky 500+ strings
    df = data_utils.data_getridofstrings(df)

    #Separate High Potential Candidates (500+ connections)
    dfHP = df[df.connection == 500]
    dfLP = df[df.connection < 10]

    #Drop Duplicates
    df = df.drop_duplicates(subset=df.columns.difference(['id'])).reset_index(drop=True)

    #Replace na's with 0
    df['fit'] = df['fit'].fillna(0)

    #Set Keys -- should be user assignable, and variable in number
    key_1 = input('First Key Word or Phrase:\n')
    key_2 = input('Second Key Word or Phrase:\n')
    key_3 = ''
    key_4 = ''
    key_5 = ''
    switch = input('Do you have any more key terms to add? (y/n):')
    if switch == 'y':
        key_3 = input('Third Key Word or Phrase: \n')
        switch = input('Do you have any more key terms to add? (y/n):')
        if switch == 'y':
            key_4 = input('Fourth Key Word or Phrase: \n')
            switch = input('Do you have any more key terms to add? (y/n):')
            if switch == 'y':
                key_5 = input('Fifth and Final Key Word or Phrase: \n')

    #consolidate keys
    search_terms = key_1 + ' ' + key_2 + ' ' + key_3 + ' ' + key_4 + ' ' + key_5
    print(search_terms + "\n Processing gensim similarity index, this may take a while..." )
    #subset df for description and job title, also separate description for use in optimization later
    df["description"] = df["job_title"] +" " + df["location"]
    documents = df["description"].tolist()
    documents_noloc = df["job_title"].tolist()

    #Preprocessing for gensim --following method from: https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32
    # unused processing steps left intact for scalability and modularity
    

    
    corpus = [data_utils.preprocess(document) for document in documents]
    query = data_utils.preprocess(search_terms)

    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus+[query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    # Compute Soft Cosine Measure between the query and the documents.
    # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
    df_glove_sorted = pd.DataFrame()
    for idx in sorted_indexes:
        print('\tID:{df.iloc[[idx], 0].to_string(index=False)} ' 
                '\n Similarity score:{doc_similarity_scores[idx]:0.3f} \
                \n Connections:{df.iloc[[idx], 3].to_string(index=False)} \
                \n Description:{documents[idx]}')
        df_glove_sorted = df_glove_sorted.append(df.iloc[[idx]], ignore_index = True)
    return(df_glove_sorted, search_terms, similarity_index)


#call main function
#df, search_terms, similarity_index = main_func("data\\raw\\potential-talents - Aspiring human resources - seeking human resources.csv")


def main_func2(df, search_terms, similarity_index):
    mask = df['fit'].values == 1
    print("Mask array :", mask)
 
    # getting non zero indices
    pos = np.flatnonzero(mask)
    print("\nRows selected :", pos)
    

    rake_nltk = Rake()
    # selecting rows
    df.iloc[pos]
    documents = df["description"].tolist()
    documents_noloc = df["job_title"].tolist()
    if pos.size != 0:
        search_terms_new = search_terms
        for i in range(len(pos)):
            rake_nltk.extract_keywords_from_text(documents_noloc[pos[i]])
            new_keys = ' '.join(rake_nltk.get_ranked_phrases())
            print(new_keys)
            search_terms_new = search_terms_new + " " + new_keys
            print(search_terms_new)
        query_new = data_utils.preprocess(search_terms_new)
        corpus = [data_utils.preprocess(document) for document in documents]
        dictionary_new = Dictionary(corpus+[query_new])
        tfidf_new = TfidfModel(dictionary=dictionary_new)

        # Create the term similarity matrix.  
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary_new, tfidf_new)
        query_tf_new = tfidf_new[dictionary_new.doc2bow(query_new)]

        index = SoftCosineSimilarity(
                tfidf_new[[dictionary_new.doc2bow(document) for document in corpus]],
                similarity_matrix)

        doc_similarity_scores_new = index[query_tf_new]
        df['scores'] = doc_similarity_scores_new
        # Output the sorted similarity scores and documents
        sorted_indexes_new = np.argsort(doc_similarity_scores_new)[::-1]
        df_glove_sorted = pd.DataFrame()
        for idx in sorted_indexes_new:
            print(f'\tID:{df.iloc[[idx], 0].to_string(index=False)} \
                    \n Similarity score:{doc_similarity_scores_new[idx]:0.3f} \
                    \n Connections:{df.iloc[[idx], 3].to_string(index=False)} \
                    \n Description:{documents[idx]}')
            df_glove_sorted = df_glove_sorted.append(df.iloc[[idx]], ignore_index = True)
        return df_glove_sorted
    else: return df



#call main function 2
#df_rough_final = main_func2(df, search_terms, similarity_index)

def clean_and_save(df, destination_path):
    df_glove_sorted_rounded = df.round({'scores': 1})
    df_revised = df_glove_sorted_rounded.sort_values(by =  ['scores', 'connection'], ascending = [False, False] , na_position = 'first')
    df_final = df_revised[['id', 'job_title', 'connection', 'location', 'fit', 'scores']]
    destination_path = "data\\processed\\sorted_list.csv"
    df_final.to_csv(destination_path)
    print('file saved successfully to ' + destination_path)
    return df_final

#df_final = clean_and_save(df_rough_final, "data\\processed\\sorted_list.csv")
