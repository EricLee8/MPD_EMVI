import os
import json
import torch
import random
import warnings
import importlib
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, ElectraTokenizerFast, DebertaV2Tokenizer
from transformers import BertConfig, ElectraConfig, DebertaV2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from utils.utils import *


accelerator = Accelerator()
if accelerator.is_local_main_process and not os.path.exists(args.save_root):
    os.mkdir(args.save_root)
accelerator.wait_for_everyone()
if accelerator.is_local_main_process and not os.path.exists(args.cache_root):
    os.mkdir(args.cache_root)
accelerator.wait_for_everyone()


ClassificationModel = importlib.import_module('models.' + args.model_file).ClassificationModel


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast),
    'roberta': (DebertaV2Config, DebertaV2Tokenizer)
}


warnings.filterwarnings("ignore")
device = accelerator.device
train_path = os.path.join(args.data_path, args.dataset, "train.json")
eval_path = os.path.join(args.data_path, args.dataset, "valid.json")
test_path = os.path.join(args.data_path, args.dataset, "test.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def cur_larger(cur_result, cur_best_result):
    for metric in METRICS:
        if cur_result[metric] != cur_best_result[metric]:
            return cur_result[metric] > cur_best_result[metric]
    return False


def prepare_inputs(batch, training=True):
    inputs = {'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'token_type_ids': batch['token_type_ids'],
                'indicator_ids': batch['indicator_ids'],
                'sep_poses': batch['sep_poses'],
                }
    if training:
        inputs.update({'labels': batch['labels']})
    return inputs


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    accelerator.print("Traning arguments:")
    accelerator.print(args)
    
    best_result = {metric: 0.0 for metric in METRICS}
    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    model, all_optimizer, train_loader, eval_dataloader, test_dataloader, scheduler = accelerator.prepare(
          model, all_optimizer, train_loader, eval_dataloader, test_dataloader, scheduler
      )

    t_total = len(train_loader) * args.epochs
    logging_interval = max(1, args.epochs // args.logging_times)

    # # DEBUG ===================================================================================================================
    # eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
    # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')
    # # DEBUG ===================================================================================================================

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            inputs = prepare_inputs(batch)
            outputs = model(**inputs)
            loss, loss_dict = outputs[0], outputs[1]
            print_loss = loss.item()
            description = "Loss:%.3f," %(print_loss)
            for k, v in loss_dict.items():
                description += f"{k}:{round(v, 3)},"
            description = description[:-1]
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            scheduler.step()
            model.zero_grad()
            pbar.set_description(description)

        if (epoch+1) % logging_interval == 0 or (epoch+1) == args.epochs:
            accelerator.print("\nEpoch {}:".format(epoch))
            eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
            # if accelerator.is_local_main_process and eval_result['f1'] < 1/3:
            #     accelerator.print("Become untrainable! Training canceled!!!")
            #     exit(1)
            if accelerator.is_local_main_process and cur_larger(eval_result, best_result):
                best_result = eval_result
            evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')
            accelerator.wait_for_everyone()


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None, cal_metrics=True):
    def _cal_metrics(pred_dict, golden_dict):
        true_labels = np.array(list(golden_dict.values()))
        pred_labels = np.array(list(pred_dict.values()))
        try:
            auc = roc_auc_score(true_labels, pred_labels)
        except ValueError:
            auc = 0.0
        # calculate threshold
        thr = 0.1
        best_thr = -1.0
        best_fc = -1.0
        while thr <= 0.9:
            current_pred_labels = (pred_labels >= thr)
            fc = f1_score(true_labels, current_pred_labels)
            if fc > best_fc:
                best_fc = fc
                best_thr = thr
            thr += 0.05
        threshold = best_thr
        current_pred_labels = (pred_labels >= threshold)
        acc = accuracy_score(true_labels, current_pred_labels)
        pre = precision_score(true_labels, current_pred_labels)
        rec = recall_score(true_labels, current_pred_labels)
        fc = (0.0 if pre == rec == 0.0 else 2 * pre * rec / (pre + rec))
        return {'auc': auc, 'f1': fc, 'acc': acc, 'recall': rec, 'precision': pre, 'threshold': thr}

    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # accelerator.wait_for_everyone()

    model.eval()
    unwrapped_model = accelerator.unwrap_model(deepcopy(model))
    prefix = eval_loader.dataset.prefix
    pred_dict = {}
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False)
            outputs = unwrapped_model(**inputs)
            preds = outputs[0]
            all_preds, all_qid_tensors = accelerator.gather((preds, batch['qid_tensor']))
            all_qids = [prefix+'-'+str(x) for x in to_list(all_qid_tensors)]
            for qid, pred in zip(all_qids, to_list(all_preds)):
                pred_dict[qid] = pred

    if accelerator.is_local_main_process and cal_metrics:
        answer_path = test_path if is_test else eval_path
        with open(answer_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        golden_dict = {}
        for idx, line in enumerate(lines):
            qid = prefix + '-' + str(idx)
            ans = json.loads(line.strip())[1]
            golden_dict[qid] = ans

        result_dict = _cal_metrics(pred_dict, golden_dict)
        accelerator.print("Test Result:" if is_test else "Eval Result:", result_dict)
        if cur_best_result is not None and cur_larger(result_dict, cur_best_result):
            accelerator.print("model and arguments saved to {}...".format(args.save_path))
            save_path = os.path.join(args.save_path, "best_model.pth")
            args_save_path = os.path.join(args.save_path, "args.pth")
            unwrapped_model = unwrapped_model.to("cpu")
            accelerator.save(unwrapped_model.state_dict(), save_path)
            accelerator.save(args, args_save_path)

    if accelerator.is_local_main_process and file_name is not None:
        with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
            json.dump(pred_dict, f, indent=2)
    del unwrapped_model
    model.train()

    return result_dict if accelerator.is_local_main_process and cal_metrics else None


def training(model, tokenizer):
    train_dataset = get_dataset(train_path, tokenizer, training=True, accelerator=accelerator)
    eval_dataset = get_dataset(eval_path, tokenizer, training=False, accelerator=accelerator)
    test_dataset = get_dataset(test_path, tokenizer, training=False, accelerator=accelerator)
    num_workers = args.cpu_count
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)

    if 'None' not in args.pretrain_path:
        accelerator.print("Loading pre-trained model from {}...".format(args.pretrain_path))
        model_saved = torch.load(args.pretrain_path)
        missing_keys, unexpected_keys = model.load_state_dict(model_saved, strict=False)
        accelerator.print("Missing keys: {}\nUnexpected_keys: {}\n".format(missing_keys, unexpected_keys))
    train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)


def inference(model, tokenizer):
    model_save_path = os.path.join(args.save_path, "best_model.pth")
    eval_dataset = get_dataset(eval_path, tokenizer, training=False, accelerator=accelerator)
    test_dataset = get_dataset(test_path, tokenizer, training=False, accelerator=accelerator)
    num_workers = args.cpu_count
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    accelerator.print("Loading trained model from {}".format(model_save_path))
    model_saved = torch.load(model_save_path)
    missing_keys, unexpected_keys = model.load_state_dict(model_saved, strict=False)
    accelerator.print("Missing keys: {}\nUnexpected_keys: {}\n".format(missing_keys, unexpected_keys))
    model, eval_dataloader, test_dataloader = accelerator.prepare(model, eval_dataloader, test_dataloader)
    accelerator.print("Evaluating on validation set...")
    evaluate(model, eval_dataloader, tokenizer, cur_best_result=None, cal_metrics=True)
    accelerator.print("Evaluating on testing set...")
    evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, cal_metrics=True)


def main():
    set_seed()

    if accelerator.is_local_main_process:
        print("Main process loading tokenizer and model...")
        tokenizer = tokenizer_class.from_pretrained(args.model_name)
        config = config_class.from_pretrained(args.model_name)
        model = ClassificationModel.from_pretrained(args.model_name, config=config)
    accelerator.wait_for_everyone() # main process downloads and caches the pre-trained models, other processes wait
    if not accelerator.is_local_main_process: # other processes load from cache
        print("Follower process {} loading tokenizer and model...\n".format(accelerator.local_process_index))
        tokenizer = tokenizer_class.from_pretrained(args.model_name)
        config = config_class.from_pretrained(args.model_name)
        model = ClassificationModel.from_pretrained(args.model_name, config=config)
    tokenizer.truncation_side = 'left'
    accelerator.wait_for_everyone()

    if args.mode == 0:
        training(model, tokenizer)
    elif args.mode == 1:
        inference(model, tokenizer)
    else:
        pass


if __name__ == "__main__":
    main()
