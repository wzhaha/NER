# -*- coding: UTF-8 -*-

import codecs

renmin_label_map = {
    'O': 'O',
    'B_nr': 'B-PER',
    'M_nr': 'I-PER',
    'E_nr': 'I-PER',
    'B_ns': 'B-LOC',
    'M_ns': 'I-LOC',
    'E_ns': 'I-LOC',
    'B_nt': 'B-ORG',
    'M_nt': 'I-ORG',
    'E_nt': 'I-ORG'
}
renmin_train_data_path = 'train_data.txt'
renmin_train_label_path = 'train_label.txt'
renmin_test_data_path = './renmin_test_data.txt'
renmin_test_label_path = './renmin_test_label.txt'


def save2file(datas, labels, save_data_path, save_label_path):
    with open(save_data_path, mode='w') as f:
        for data in datas:
            data_str = " ".join(str(i) for i in data)
            f.write(data_str+'\n')  # write 写入

    with open(save_label_path, mode='w') as f:
        for label in labels:
            label_str = " ".join(str(i) for i in label)
            f.write(label_str+'\n')  # write 写入


def renmin_data_prepare():
    # 训练集与测试集的比例为3:1
    rate = 0.75
    datas = list()
    labels = list()
    input_data = codecs.open('renmin_original.txt', 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.split()
        linedata=[]
        linelabel=[]
        numNotO=0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(renmin_label_map[word[1]])
            if word[1] != 'O':
                numNotO += 1
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    data_len = len(datas)
    data_slice_index = int(data_len*rate)
    # 保存到文件
    save2file(datas[:data_slice_index], labels[:data_slice_index], renmin_train_data_path, renmin_train_label_path)
    save2file(datas[data_slice_index:], labels[data_slice_index:], renmin_test_data_path, renmin_test_label_path)
    print('人民日报1998 数据长度：', len(datas))
    print('人民日报1998 训练集长度：', len(labels[:data_slice_index]))
    print('人民日报1998 测试集长度：', len(labels[data_slice_index:]))


if __name__ == '__main__':
    renmin_data_prepare()
