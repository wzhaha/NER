import os
import sys
sys.path.insert(0, "src")
from config import *
import logging
from pre_processor import processors, load_and_cache_examples, ner_F1
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
from eval import *
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logging.getLogger().setLevel(logging.INFO)


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) * config.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(config.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(config.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(config.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(config.model_name_or_path, "scheduler.pt")))

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", config.epochs)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    set_seed()  # Added here for reproductibility
    for epoch_num in range(int(config.epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()
            tr_loss += loss.item()
            # # 梯度截断
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                logging.info("epoch_num = {} global_step = {}, loss = {}".format(epoch_num+1, global_step, loss.item()))
        if config.evaluate_during_training:
            results = evaluate(config, model, tokenizer)
            for key, value in results.items():
                logging.info("global_step = {}, eval_{} = {}".format(global_step, key, value))

            # Save model checkpoint
            output_dir = os.path.join(config.save, "bert_softmax_checkpoint-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info("Saving model checkpoint to %s", output_dir)

            torch.save(config, os.path.join(output_dir, "training_args.bin"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logging.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(config, model, tokenizer, prefix=""):
    eval_output_dir = config.save
    results = {}
    eval_dataset, eval_examples = load_and_cache_examples(config, config.use_data, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size)

    # Eval!
    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", config.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    masks = None
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            masks = inputs["attention_mask"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    # ner: logits=scores .shape=[examp_nums, seq_len, num_labels]
    preds = np.argmax(preds, axis=2)
    result = ner_F1(preds, out_label_ids, masks)

    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logging.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def test(config, model, tokenizer, label_list):
    eval_output_dir = config.test_out_path

    eval_dataset, eval_examples = load_and_cache_examples(config, config.use_data, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size)

    # Eval!
    logging.info("***** Running test *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", config.batch_size)


    id2label_map = {i: label for i, label in enumerate(label_list)}
    id2label_map[0] = 0

    # 模型预测的全部结果
    model_predict = []
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch_index, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        masks = inputs["attention_mask"].detach().cpu().numpy()
        batch_examples = eval_examples[batch_index*config.batch_size:(batch_index+1)*config.batch_size]
        for i in range(len(preds)):
            sent_res = []
            num = sum(masks[i]) - 2
            pres = [id2label_map[i] for i in preds[i][1: 1 + num]]
            ground_label = [id2label_map[i] for i in out_label_ids[i][1: 1 + num]]
            example = batch_examples[i].text_a
            for j in range(len(pres)):
                sent_res.append([example[j], ground_label[j], pres[j]])
            model_predict.append(sent_res)
        if batch_index%50 == 0:
            print('{}/{}'.format(batch_index+1, len(eval_dataloader)))
    label_path = os.path.join(eval_output_dir, 'test_label')
    metric_path = os.path.join(eval_output_dir, 'test_metric')
    for line in conlleval(model_predict, label_path, metric_path):
        print(line)


def main():
    # config parameter
    config = Config()
    config.print_parameters()

    processor = processors[config.use_data]()
    label_list = processor.get_labels(config)
    num_labels = len(label_list)
    config_class, model_class, tokenizer_class = BertConfig, BertForTokenClassification, BertTokenizer

    bert_config = config_class.from_pretrained(
        config.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config.use_data
    )
    tokenizer = tokenizer_class.from_pretrained(
        config.model_name_or_path,
        # do_lower_case=config.do_lower_case,
    )
    model = model_class.from_pretrained(
        config.model_name_or_path,
        from_tf=bool(".ckpt" in config.model_name_or_path),
        config=bert_config,
    )

    model.to(config.device)

    if config.mode == 'train':
        train_dataset, train_examples = load_and_cache_examples(config, config.use_data, tokenizer, evaluate=False)
        train(config, train_dataset, model, tokenizer)

    if config.mode == 'test':
        checkpoint = config.test_checkpoint
        model = model_class.from_pretrained(checkpoint)
        model.to(config.device)
        test(config, model, tokenizer, label_list)


if __name__ == "__main__":
    main()