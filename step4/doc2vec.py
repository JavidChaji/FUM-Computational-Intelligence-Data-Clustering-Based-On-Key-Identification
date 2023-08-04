import gensim
import pandas as pd
from nltk.tokenize import word_tokenize


def doc2Vec(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).replace("', '", " ")

    text_array = [gensim.models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text_array)] 
    documents_tags_dataframe = pd.DataFrame(text_array)
    documents_tags_dataframe.to_csv('step4/documents_tags.csv', index=False)
    
    model = gensim.models.Doc2Vec(dm=1, vector_size=65, hs=1, min_count=2, sample = 12000,window=3, alpha=0.025, min_alpha=0.00025)
    model.build_vocab(text_array)
    model.train(text_array, total_examples=model.corpus_count, epochs=70)
    model.save('step4/doc2vec.model')

    return text_array


tokens_with_lemmatizing_csv = pd.read_csv('step3/three_tokens_with_lemmatizing.csv')

extracted_key_tokens = doc2Vec(tokens_with_lemmatizing_csv['Comment'])

extracted_key_tokens_dataframe = pd.DataFrame(extracted_key_tokens)
extracted_key_tokens_dataframe.to_csv('step4/four_point_one_extracted_key_tokens.csv', index=False)
