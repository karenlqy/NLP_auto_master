import os
import jieba
import time
import pandas as pd

if __name__ == '__main__':
    root_path = os.path.abspath('../')
    train_path = os.path.join(root_path, 'data/AutoMaster_TrainSet.csv')
    test_path = os.path.join(root_path, 'data/AutoMaster_TestSet.csv')
    stopwords_path = os.path.join(root_path, 'data/stop_words.txt')
    write_path = os.path.join(root_path, 'data/corpus_w2v.txt')

    with open(train_path, 'r', encoding = 'utf-8') as f_read:
        data1 = f_read.readlines()
        f_read.close()
    with open(test_path, 'r', encoding = 'utf-8') as f_read2:
        data2 = f_read2.readlines()
        f_read.close()

    stopwords = [t.strip() for t in open(stopwords_path, 'r', encoding = 'utf-8').readlines()]

#remove stopwords and create corpus
    with open(write_path, 'w', encoding='utf-8') as f_write:
        temp = []
        for line in data1:
            line = list(jieba.cut(str(line).strip()))
            for word in line:
                if word not in stopwords:
                    temp.append(word)
        f_write.write(' '.join(temp))

        temp2 = []
        for line in data2:
            line = list(jieba.cut(str(line).strip()))
            for word in line:
                if word not in stopwords:
                    temp2.append(word)
        f_write.write(' '.join(temp2))
        f_write.close()
