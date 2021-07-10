# Bert NER

使用Bert对人民日报语料库进行了命名实体识别。

## 使用环境

pytorch

## 项目结构

+ checkpoints： 存放训练好的模型
+ dataset：存放处理好的数据集，其中包含了处理程序
+ model：模型实现
+ result：评估结果输出
+ main.py：主程序入口
+ eval.py：性能评估

## 运行

运行 `pip install -r requirements.txt`安装运行所需的库

修改main中config类中的mode以及其他参数进行训练和测试。

