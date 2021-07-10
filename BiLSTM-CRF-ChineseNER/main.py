import torch
from data_read import read_dictionary
import numpy as np
from util import *


class Config():
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.seed = 1111
        self.use_cuda = True if torch.cuda.is_available() else False
        self.lr = 0.001
        self.use_crf = True
        self.mode = 'test'
        self.save = './checkpoints/lstm_crf.pth'
        self.data = 'dataset'
        self.use_data = 'renmin'
        self.word_ebd_dim = 300
        self.dropout = 0.5
        self.lstm_hsz = 300
        self.lstm_layers = 2
        self.l2 = 0.005
        self.clip = 0.5
        self.result_path = './result'


config = Config()

torch.manual_seed(config.seed)

# load data
from data_loader import DataLoader
from data_read import read_corpus, tag2label
import os
from eval import conlleval

# 读取数据
# data1
if config.use_data == 'default':
    source_data_path = './dataset/source_data.txt'
    source_label_path = './dataset/source_label.txt'
    test_data_path = './dataset/test_data.txt'
    test_label_path = './dataset/test_label.txt'
    word2id_path = './dataset/vocab.pkl'

# data2
if config.use_data == 'renmin':
    source_data_path = './dataset/renMinRiBao/renmin_train_data.txt'
    source_label_path = './dataset/renMinRiBao/renmin_train_label.txt'
    test_data_path = './dataset/renMinRiBao/renmin_test_data.txt'
    test_label_path = './dataset/renMinRiBao/renmin_test_label.txt'
    word2id_path = './dataset/renMinRiBao/renmin_vocab.pkl'

# data3
if config.use_data == 'msra':
    source_data_path = './dataset/MSRA/msra_train_data.txt'
    source_label_path = './dataset/MSRA/msra_train_label.txt'
    test_data_path = './dataset/MSRA/msra_test_data.txt'
    test_label_path = './dataset/MSRA/msra_test_label.txt'
    word2id_path = './dataset/MSRA/msra_vocab.pkl'

sents_train, labels_train, config.word_size, _ = read_corpus(source_data_path, source_label_path, word2id_path)
sents_test, labels_test, _, test_data_origin = read_corpus(test_data_path, test_label_path, word2id_path,
                                                           is_train=False)
config.label_size = len(tag2label)

train_data = DataLoader(sents_train, labels_train, cuda=config.use_cuda, batch_size=config.batch_size)
test_data = DataLoader(sents_test, labels_test, cuda=config.use_cuda, shuffle=False, evaluation=True,
                       batch_size=config.batch_size)

from model import Model

model = Model(config)

if config.use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=config.l2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def train(epoch):
    model.train()
    total_loss = 0
    running_loss = 0
    batch_idx = 0
    for word, label, seq_lengths, _ in train_data:
        batch_idx = batch_idx + 1
        optimizer.zero_grad()
        loss, _ = model(word, label, seq_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        running_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(label), train_data.sents_size,
                       100. * batch_idx / train_data._stop_step, running_loss / 50))
            running_loss = 0

    return total_loss / train_data._stop_step


def evaluate():
    model.eval()
    eval_loss = 0

    model_predict = []

    label2tag = {}
    for tag, lb in tag2label.items():
        label2tag[lb] = tag if lb != 0 else lb

    label_list = []

    for word, label, seq_lengths, unsort_idx in test_data:
        loss, _ = model(word, label, seq_lengths)
        pred = model.predict(word, seq_lengths)
        pred = pred[unsort_idx]
        seq_lengths = seq_lengths[unsort_idx]

        for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
            pred_ = list(pred[i][:seq_len].cpu().numpy())
            label_list.append(pred_)

        eval_loss += loss.detach().item()

    for label_, (sent, tag) in zip(label_list, test_data_origin):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if len(label_) != len(sent):
            print(len(sent))
            print(len(label_))
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)

    label_path = os.path.join(config.result_path, 'label')
    metric_path = os.path.join(config.result_path, 'result_metric')

    for line in conlleval(model_predict, label_path, metric_path):
        print(line)

    return eval_loss / test_data._stop_step


def test(model_path, data, origin_data):
    net_data = torch.load(model_path, map_location="cpu")
    model.load_state_dict(net_data)
    model.eval()
    model_predict = []

    label2tag = {}
    for tag, lb in tag2label.items():
        label2tag[lb] = tag if lb != 0 else lb

    label_list = []

    for word, label, seq_lengths, unsort_idx in data:
        pred = model.predict(word, seq_lengths)
        pred = pred[unsort_idx]
        seq_lengths = seq_lengths[unsort_idx]

        for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
            pred_ = list(pred[i][:seq_len].cpu().numpy())
            label_list.append(pred_)

    for label_, (sent, tag) in zip(label_list, origin_data):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)

    label_path = os.path.join(config.result_path, 'test_label')
    metric_path = os.path.join(config.result_path, 'test_metric')

    for line in conlleval(model_predict, label_path, metric_path):
        print(line)


def test_sentense(model_path, sentense):
    net_data = torch.load(model_path, map_location="cpu")
    model.load_state_dict(net_data)
    model.eval()

    word2id = read_dictionary(word2id_path)

    label2tag = {}
    for tag, lb in tag2label.items():
        label2tag[lb] = tag if lb != 0 else lb

    sentence_id = []
    for word in sentense:
        # 训练与测试通用的处理
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'

        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    inst_data_tensor = torch.from_numpy(np.asarray(sentence_id))
    inst_data_tensor = inst_data_tensor.unsqueeze(0)
    seq_len = torch.LongTensor([len(sentence_id)])
    pred = model.predict(inst_data_tensor, seq_len)
    pred_ = list(pred[0][:seq_len].cpu().numpy())
    tag = [label2tag[label__] for label__ in pred_]

    # 句子中包含的实体及其类型
    ners = []
    ners_type = []
    temp = ''
    for index, char in enumerate(sentense):
        if tag[index] != 0:
            if tag[index][0] == 'B':
                ners_type.append(tag[index][2:])
                if temp != '':
                    ners.append(temp)
                    temp = ''
            temp += char
        if index == len(sentense) - 1 and temp != '':
            ners.append(temp)
    print('sentense: ' + sentense)
    for (item, item_type) in zip(ners, ners_type):
        print(item + ': ', item_type)


import time

if config.mode == 'train':
    best_acc = None
    total_start_time = time.time()
    train_loss = []
    print('-' * 90)
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        loss = train(epoch)
        scheduler.step()
        train_loss.append(loss)
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
            epoch, time.time() - epoch_start_time, loss))
        eval_loss = evaluate()
        torch.save(model.state_dict(), config.save)

    draw_train_process('loss', train_loss, './loss.png')

if config.mode == 'test':
    model_path = 'checkpoints/renmin_300_300_2.pth'
    test(model_path, test_data, test_data_origin)