# Bert NER
使用Bert对人民日报语料库进行了命名实体识别。

## 使用环境
pytorch

## 项目结构

+ checkpoints： 存放训练好的模型
+ chinese_wwm_pytorch：开源项目[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)提供的Bert-Chinese中文预训练模型。
+ dataset：存放处理好的数据集，其中包含了处理程序
+ model：模型定义，这里只定义了Bert+Crf，Bert+Softmax直接使用提供的BertForTokenClassification
+ result：评估结果输出
+ src：bert源代码
+ bert_crf_main.py：Bert+Crf的实现主程序入口
+ bert_softmax_main.py：Bert+Softmax的实现主程序入口
+ config.py：配置文件，运行代码时需要视情况修改
+ eval.py：性能评估
+ pre_processor.py：对输入Bert的数据进行预处理

## 运行

从https://github.com/ymcui/Chinese-BERT-wwm下载预训练的Chinese-wwm模型放到根目录下。

运行 `pip install -r requirements.txt`安装运行所需的库

修改config中的mode以及其他参数进行训练和测试。

