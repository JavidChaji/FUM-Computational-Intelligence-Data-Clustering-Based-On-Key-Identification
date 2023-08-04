import re
import nltk 
import pandas as pd


def tokenize(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = text_array.loc[i].replace('\\n',' ')
            text_array.loc[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', text_array.loc[i], flags=re.MULTILINE) 
            text_array.loc[i] = re.sub(r"[0-9]+", " ", text_array.loc[i])
            text_array.loc[i] = re.sub(r"[^a-zA-Z0-9]", " ", text_array.loc[i].lower())
            text_array.loc[i] = nltk.word_tokenize(text_array.loc[i])
            
    return text_array

data_csv = pd.read_csv('dataset/train.csv')

pure_tokens = tokenize(data_csv['Comment'])

pure_tokens_dataframe = pd.DataFrame(pure_tokens)
pure_tokens_dataframe.to_csv('./one_pure_tokens.csv', index=False)
