
import numpy as np
import pandas as pd
import os
import jieba
import re
import copy
import codecs
import time
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence



root_path = os.path.abspath('../')
stopwords_path = os.path.join(root_path, 'data/stop_words.txt')

stopwords = [t.strip() for t in open(stopwords_path, 'r', encoding = 'utf-8').readlines()]


#split word
def seg_line(line):
    tokens = jieba.cut(str(line), cut_all = False)
    words = []
    for word in tokens:
        words.append(word)
    return " ".join(words)

#data cleaning, align data length
def split_data(train_path, test_path, train_save_path, test_save_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)


    for k in ['Question','Dialogue','Report']:
        for i in range(len(train[k])):
            line = train[k].get(i)
            #替换掉 车主说|技师说|语音|图片|你好|您好等词语
            line = re.sub(u"[\u6280\u5e08\u8bf4 | \u8f66\u4e3b\u8bf4 | \u8bed\u97f3 | \u56fe\u7247 |\u4f60\u597d |\u60a8\u597d]", '', line)
            line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', str(line))
            line = ''.join(line) #str(line)
            train[k][i] = seg_line(line)

    for k in ['Question','Dialogue']:
        for i in range(len(test[k])):
            line = test[k].get(i)
            #替换掉 车主说|技师说|语音|图片|你好|您好等词语
            line = re.sub(u"[\u6280\u5e08\u8bf4 | \u8f66\u4e3b\u8bf4 | \u8bed\u97f3 | \u56fe\u7247 |\u4f60\u597d |\u60a8\u597d]", '', line)
            line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', str(line))
            line = ''.join(line) #str(line)
            test[k][i] = seg_line(line)

    train.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    train['input'] =  train['Dialogue'] + train['Question']
    test['input'] =  test['Dialogue'] + test['Question']

    train.drop(['Brand', 'Model', 'Dialogue', 'Question'], axis = 1, inplace = True)
    test.drop(['Brand', 'Model', 'Dialogue', 'Question'], axis = 1, inplace = True)

    train.to_csv(train_save_path, index=False, encoding = 'utf-8')
    test.to_csv(test_save_path, index=False, encoding = 'utf-8')


def stat_dict(lines):
    word_dict = {}
    for line in lines:
        tokens = str(line).split(" ")
        for t in tokens:
            t = t.strip()
            if t:
                word_dict[t] = word_dict.get(t,0) + 1
    return word_dict

def filter_dict(word_dict, min_count=3):
    out_dict = {}
    keys = word_dict.keys()
    for k in keys:
        if word_dict[k] >= min_count:
            out_dict[k] = word_dict[k]
    return out_dict



def build_vocab(lines, min_count=3):
    start_token = u"<s>"
    end_token = u"<e>"
    unk_token = u"<unk>"
    word_dict = stat_dict(lines)
    word_dict = filter_dict(word_dict, min_count)

    sorted_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_words = [w for w,c in sorted_dict]
    sorted_words = [start_token, end_token, unk_token] + sorted_words
    vocab = dict([(w,i) for i,w in enumerate(sorted_words)])
    reverse_vocab = dict([(i,w) for i,w in enumerate(sorted_words)])

    return vocab, reverse_vocab

def save_vocab(vocab, path):
    output = codecs.open(path, "w", "utf-8")
    for w,i in sorted(vocab.items(), key=lambda x:x[1]):
        output.write("%s %d\n" %(w,i))
    output.close()

def get_model_from_file(model_path):
    # model = KeyedVectors.load('w2v_gensim', mmap='r')
    model = Word2Vec.load(model_path)
    return model

def prepare_padding(infile, model, outfile, oov_path):
    '''add <start> <end> <pad> <unk> token, prepare sample with right length and retrain word2vec'''
    train = pd.read_csv(infile, encoding='utf-8')
    lines = []

    lines.extend(list(train['input'].values))
    max_lens = 0
    new_lines = []
    oov_list = []

    for line in lines:
        if len(line) > max_lens:
            max_lens = len(line)+2
        else:
            max_lens = max_lens


        new_word_list = ['<s>']
        word_list = line.strip().split(' ')
        for word in word_list:
            if word in model.wv.vocab:
                new_word_list.append(word)
            else:
                new_word_list.append('<unk>')
                oov_list.append(word)
        new_word_list.append('<e>')
        # print(new_word_list)
        newline = ' '.join(new_word_list)
        # print(newline)
        new_lines.append(newline)

    with open(outfile, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)
            f.write('\n')
    with open(oov_path, 'w', encoding='utf-8') as f:
        for oov in oov_list:
            f.write(' '.join(oov))
            f.write('\n')
    f.close()
    return max_lens

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





if __name__ == "__main__":
    root_path = os.path.abspath('../')
    train_path = os.path.join(root_path, 'data/train_5k.csv')
    test_path = os.path.join(root_path, 'data/AutoMaster_TestSet.csv')
    train_save_path = os.path.join(root_path, 'data/train_save.csv')
    test_save_path = os.path.join(root_path, 'data/test_save.csv')
    vocab_path = os.path.join(root_path, "data/vocab_dictionary.txt")
    corpus_path = os.path.join(root_path, "data/corpus_w2v.txt")
    model_path = os.path.join(root_path, "data/w2v.model")
    train_padded_path = os.path.join(root_path, "data/train_padded_save.txt")
    oov_path = os.path.join(root_path, "data/oov_dict.txt")
    min_count = 3

    split_data(train_path, test_path, train_save_path, test_save_path)

    train = pd.read_csv(train_save_path, encoding='utf-8')

    lines = []
    for k in ['input', 'Report']:
        lines.extend(list(train[k].values))

    vocab, reverse_vocab = build_vocab(lines, min_count)

    save_vocab(vocab, vocab_path)
    train_w2v_model(corpus_path,model_path)

    model = get_model_from_file(model_path)


for line in lines[:5]:
    word_list = line.split()
word_list

for line in lines[:5]:
    word_list = line.split()
    for word in word_list:
        if word in model.wv.vocab:
            new_word_list.append(word)
        else:
            new_word_list.append('<unk>')
    new_word_list.append('<e>')
