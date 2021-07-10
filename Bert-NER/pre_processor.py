import csv
import os
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import json
import logging
from src.transformers.tokenization_utils import PreTrainedTokenizer
import torch
from torch.utils.data import TensorDataset


logging.getLogger().setLevel(logging.INFO)


@dataclass
class NERInputExample:

    guid: str
    text_a: str
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class InputExample:

    guid: str
    text_a: str
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[Union[int, float]]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_data(cls, data_path, mode):
        """Reads a tab separated value file."""
        if mode == 'train':
            with open(os.path.join(data_path, "train_data.txt"), encoding='utf-8') as fr:
                datas = fr.readlines()
            with open(os.path.join(data_path, "train_label.txt")) as fl:
                labels = fl.readlines()
        else:
            with open(os.path.join(data_path, "test_data.txt"), encoding='utf-8') as fr:
                datas = fr.readlines()
            with open(os.path.join(data_path, "test_label.txt")) as fl:
                labels = fl.readlines()
        return datas, labels


class RenminNerProcessor(DataProcessor):

    def get_train_examples(self, config):
        """See base class."""
        datas, labels = self._read_data(os.path.join(config.data_path, config.use_data), mode='train')
        examples = []
        for (i, line) in enumerate(zip(datas, labels)):
            guid = "%s-%s" % ("train", i)
            text_a = line[0].replace("\n", "").replace(" ", "")
            label = line[1].replace("\n", "").split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a,  label=label))
        return examples

    def get_test_examples(self, config):
        """See base class."""
        datas, labels = self._read_data(os.path.join(config.data_path, config.use_data), mode='test')
        examples = []
        for (i, line) in enumerate(zip(datas, labels)):
            guid = "%s-%s" % ("test", i)
            text_a = line[0].replace("\n", "").replace(" ", "")
            label = line[1].replace("\n", "").split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, config):
        """See base class."""
        labels = []
        with open(os.path.join(config.data_path, config.use_data, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


processors = {
    'renmin': RenminNerProcessor
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int],
        label_list: List[str],
):
    def convert_text_to_ids(text):

        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))
        attention_mask = [1] * text_len + [0] * (max_length - text_len)
        token_type_ids = [0] * max_length

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids


    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for i in range(len(examples)):

        tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids(examples[i].text_a)

        label_id = [label_map["O"]]
        for j in range(len(tokens1) - 2):
            label_id.append(label_map[examples[i].label[j]])
        label_id.append(label_map["O"])
        if len(label_id) < max_length:
            label_id = label_id + [label_map["O"]] * (max_length - len(label_id))

        feature = InputFeatures(
            tokens1=tokens1,
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            label_id=label_id)

        features.append(feature)

    return features


def load_and_cache_examples(config, use_data, tokenizer, evaluate=False):
    processor = processors[use_data]()
    logging.info("Creating features from dataset file at %s", use_data)
    label_list = processor.get_labels(config)
    if evaluate:
        examples = (
            processor.get_test_examples(config)
        )
    else:
        examples = (
            processor.get_train_examples(config)
        )
    features = convert_examples_to_features(
        examples, tokenizer, max_length=config.max_seq_length, label_list=label_list
    )

    # Convert to Tensors and build dataset
    all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
    all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels)

    return dataset, examples


def ner_F1(preds, labels, mask_indicators):
    assert len(preds) == len(labels) == len(mask_indicators)
    total_preds = []
    total_ground = []
    for i in range(len(preds)):
        num = sum(mask_indicators[i]) - 2
        total_preds.extend(preds[i][1: 1 + num])
        total_ground.extend(labels[i][1: 1 + num])

    refer_label = total_ground
    pred_label = total_preds
    fn = dict()
    tp = dict()
    fp = dict()
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:
            if refer_label[i] not in tp:
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1
        else:
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0
            fp[pred_label[i]] += 1
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0
            fn[refer_label[i]] += 1
    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    p_total = float(tp_total) / (tp_total + fp_total)
    r_total = float(tp_total) / (tp_total + fn_total)
    f_micro = 2 * p_total * r_total / (p_total + r_total)

    return {"f1_score": f_micro}