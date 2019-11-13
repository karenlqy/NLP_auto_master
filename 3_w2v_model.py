import jieba
import os
import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

def get_stop_words(file: str, encoding='utf-8'):
    ret = [l for l in open(file, encoding=encoding).read()]
    return ret



def write_token_to_file(infile, outfile):
    words = []
    for line in open(infile, 'r', encoding='utf-8'):
        line = line.strip()
        if line:
            w = jieba.lcut(line)
            words += w + ['\n']
    outfile.writelines(' '.join(words))


def train_w2v_model(txtPath,model_path):
    start_time = time.time()
    w2v_model = Word2Vec(LineSentence(txtPath), workers=4)
    w2v_model.save(model_path) # Can be used for continue trainning
    # w2v_model.wv.save('w2v_gensim') # Smaller and faster but can't be trained later
    print('elapsed time:', time.time() - start_time)


def get_model_from_file(model_path):
    # model = KeyedVectors.load('w2v_gensim', mmap='r')
    model = Word2Vec.load(model_path)
    return model

if __name__ == "__main__":
    root_path = os.path.abspath('../')
    txt_path = os.path.join(root_path, "data/corpus_w2v.txt")
    model_path = os.path.join(root_path, "data/w2v.model")
    train_w2v_model(txt_path,model_path)
    # model = get_model_from_file(model_path)
    # words_list = model.vocabulary
    # print(model.corpus_total_words)
    # print(words_list)
    # n = 0
    # for word in words_list:
    #     n +=1
    #
    # print(n)
    #
    # if word not in words_list:
    #     print()
    # print(model['福特福克斯手动'])
    #
