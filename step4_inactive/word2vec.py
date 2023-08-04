import gensim
import pandas as pd


def word2Vec(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).replace("', '", " ")
    text_array = text_array.apply(gensim.utils.simple_preprocess)
    # print(text_array)
    model = gensim.models.Word2Vec(window=3, min_count=3, workers=4)
    model.build_vocab(text_array, progress_per=10000)
    model.train(text_array, total_examples=model.corpus_count, epochs=70)
    model.save('word2vec.model')

    print(model.wv.get_vector('good'))

    # for i in range(len(text_array)):
    #     if type(text_array.loc[i]) != float:
    #         text_array.loc[i] 
    #         print(str((i/13741)*100)+"%\r")
    return text_array


tokens_with_lemmatizing_csv = pd.read_csv('step3/three_tokens_with_lemmatizing.csv')

extracted_key_tokens = word2Vec(tokens_with_lemmatizing_csv['Comment'])

extracted_key_tokens_dataframe = pd.DataFrame(extracted_key_tokens)
extracted_key_tokens_dataframe.to_csv('./four_extracted_key_tokens.csv', index=False)
