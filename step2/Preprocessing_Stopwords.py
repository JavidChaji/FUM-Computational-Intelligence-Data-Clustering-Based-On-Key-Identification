import nltk 
import pandas as pd


def stopword_remover(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).split("', '")
    for i in range(len(text_array)):
        if type(text_array.loc[i]) != float:
            text_array.loc[i] = [word for word in text_array.loc[i] if word not in nltk.corpus.stopwords.words('english')]
    return text_array


pure_tokens_csv = pd.read_csv('step1/one_pure_tokens.csv')

tokens_without_stopwords = stopword_remover(pure_tokens_csv['Comment'])

tokens_without_stopwords_dataframe = pd.DataFrame(tokens_without_stopwords)
tokens_without_stopwords_dataframe.to_csv('./two_tokens_without_stopwords.csv', index=False)
