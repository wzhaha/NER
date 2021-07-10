import torch


class Config():
    def __init__(self):
        self.batch_size = 8
        self.epochs = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-5
        self.weight_decay = 0.01
        self.crf_learning_rate = 1e-3
        self.max_seq_length = 50
        self.logging_steps = 5
        self.evaluate_during_training = True
        self.mode = 'train'
        self.test_checkpoint = './checkpoints/bert_softmax_checkpoint-5'
        self.test_out_path = './result'
        self.save = './checkpoints/'
        self.data_path = './dataset'
        self.use_data = 'renmin'
        self.model_name_or_path = 'chinese_wwm_pytorch'

    def print_parameters(self):
        print('Config parameters')
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))