from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from gensim.models import Word2Vec
import os
import pandas as pd
from itertools import islice

def load_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

max_features = 300
maxlen = 300
embed_size = 100
max_length_inp,max_length_targ= 500, 50

def tokenize(lang,max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=False,num_words = max_features)
    tokenizer.fit_on_texts(lang) #这样保证encoder和decoder使用的是同一个字典，按道理讲，decoder输出少，不应该维护那么长的字典
    word_index = tokenizer.word_index
    # print(word_index)
    #print(len(word_index))#一共十多万个词语，太多了
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post',maxlen=max_len)
    return tensor, tokenizer,word_index


def load_dataset(data):
    source = [str(m) for m in data['input'].values.tolist()]
    target = [str(m) for m in data['Report'].values.tolist()]
    # inp_lang = data['Report'].values.tolist()[:num_examples]
    input_tensor, tokenizer1,word_index1 = tokenize(source,max_length_inp)
    target_tensor, tokenizer2,word_index2 = tokenize(target,max_length_targ)
    return input_tensor, target_tensor,word_index1,word_index2,tokenizer1,tokenizer2

def get_embedding():
    root_path = os.path.abspath('../')
    model_path = os.path.join(root_path, "data/w2v.model")
    data_path = os.path.join(root_path, "data/train_save.csv")
    data = pd.read_csv(data_path, encoding='utf-8')
    model = load_w2v_model(model_path)
    input_tensor, target_tensor,word_index1,word_index2,tokenizer1,tokenizer2= load_dataset(data)

    #encoder embedding
    nb_words1 = min(max_features,len(word_index1))
    embedding_matrix1 = np.zeros((nb_words1, embed_size))
    # print(len(embedding_matrix[2]))
    for word,i in word_index1.items():
        if int(i) >= nb_words1: continue
        if word not in model.wv.vocab:
            embedding_vector = np.random.uniform(-0.025, 0.025, (embed_size))
            embedding_matrix1[i] = embedding_vector
        else:
            embedding_vector = model.wv[word]
            embedding_matrix1[i] = embedding_vector
        # print(word)
        # print(embedding_matrix1[i])

    # decoder embedding
    nb_words2 = min(max_features, len(word_index2))
    embedding_matrix2 = np.zeros((nb_words2, embed_size))
    # print(len(embedding_matrix[2]))
    for word, i in word_index2.items():
        if int(i) >= nb_words2: continue
        if word not in model.wv.vocab:
            embedding_vector = np.random.uniform(-0.025, 0.025, (embed_size))
            embedding_matrix2[i] = embedding_vector
        else:
            embedding_vector = model.wv[word]
            embedding_matrix2[i] = embedding_vector
        # print(word)
        # print(embedding_matrix2[i])
    return embedding_matrix1,embedding_matrix2,input_tensor,target_tensor,tokenizer1,tokenizer2


if __name__ == "__main__":
    root_path = os.path.abspath('../')
    model_path = os.path.join(root_path, "data/train_save.csv")
    data = pd.read_csv(model_path,encoding='utf-8')
    embedding_matrix1,embedding_matrix2,input_tensor,target_tensor,tokenizer1,tokenizer2 = get_embedding()
    print(embedding_matrix1.shape)
    print(embedding_matrix2.shape)
    print(len(input_tensor))
    print(len(target_tensor))
    print(len(input_tensor[2]))
    print(len(target_tensor[2]))
