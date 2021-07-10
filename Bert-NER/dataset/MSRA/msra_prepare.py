# coding:utf-8
import codecs

msra_label_map = {
    'O': 'O',
    'nr': '-PER',
    'ns': '-LOC',
    'nt': '-ORG',
}
msra_train_ori = './msra_original_train.txt'
msra_test_ori = './msra_original_test.txt'
msra_train_data_path = './msra_train_data.txt'
msra_train_label_path = './msra_train_label.txt'
msra_test_data_path = './msra_test_data.txt'
msra_test_label_path = './msra_test_label.txt'


def save2file(datas, labels, save_data_path, save_label_path):
    with open(save_data_path, mode='w') as f:
        for data in datas:
            data_str = " ".join(str(i) for i in data)
            f.write(data_str+'\n')  # write 写入

    with open(save_label_path, mode='w') as f:
        for label in labels:
            label_str = " ".join(str(i) for i in label)
            f.write(label_str+'\n')  # write 写入


def msra_data_prepare(ori_data_path, data_path, label_path):
    datas = list()
    labels = list()
    input_data = codecs.open(ori_data_path, 'r', 'utf-8')

    for line in input_data.readlines():
        line = line.strip().split()
        linedata = []
        linelabel = []

        if len(line) == 0:
            continue
        for word in line:
            word = word.split('/')
            linedata.extend(list(word[0]))
            if word[1] != 'o':
                if len(word[0]) == 1:
                    linelabel.append('B'+msra_label_map[word[1]])
                elif len(word[0]) == 2:
                    linelabel.append('B' + msra_label_map[word[1]])
                    linelabel.append('I' + msra_label_map[word[1]])
                else:
                    linelabel.append('B' + msra_label_map[word[1]])
                    for j in word[0][1:len(word[0])]:
                        linelabel.append('I' + msra_label_map[word[1]])
            else:
                for j in word[0]:
                    linelabel.append('O')
        datas.append(linedata)
        labels.append(linelabel)
    input_data.close()
    save2file(datas, labels, data_path, label_path)


if __name__ == '__main__':
    msra_data_prepare(msra_train_ori, msra_train_data_path, msra_train_label_path)
    msra_data_prepare(msra_test_ori, msra_test_data_path, msra_test_label_path)
