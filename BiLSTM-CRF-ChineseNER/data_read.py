import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "START": 7, "STOP": 8
             }


def read_corpus(corpus_path, label_path, word2id_path, is_train=True):
    sentences = []
    labels = []
    data = []
    
    with open(corpus_path, encoding='utf-8') as fr:
        lines_co = fr.readlines()
    with open(label_path) as fl:
        lines_lb = fl.readlines()
    
    if not is_train:
        word2id = read_dictionary(word2id_path)
    else:
        word2id = {}

    for line_co, line_lb in zip(lines_co, lines_lb):
        sent_ = line_co.strip().split()
        tag_ = line_lb.strip().split()
        data.append((sent_, tag_))
        
        sentence_id = []
        for word in sent_:
            # 训练与测试通用的处理
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'

            if is_train:
                if word not in word2id:
                    word2id[word] = len(word2id)+1
            else:
                if word not in word2id:
                    word = '<UNK>'

            sentence_id.append(word2id[word])

        label_id = []
        for tag in tag_:
            label = tag2label[tag]
            label_id.append(label)

        sentences.append(sentence_id)
        labels.append(label_id)

    # 填充词典<UNK>和<PAD>并保存到文件
    if is_train:
        word2id['<UNK>'] = len(word2id)+1
        word2id['<PAD>'] = 0

        print('vocabulary length:', len(word2id))
        with open(word2id_path, 'wb') as fw:
            pickle.dump(word2id, fw)

    return sentences, labels, len(word2id), data


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))
    return word2id



