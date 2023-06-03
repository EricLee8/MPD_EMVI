import os
import json
import torch
import numpy as np
import random
import warnings
import importlib
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import BertTokenizer, EncoderDecoderModel
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.eval import COCOEvalCap
from utils.config import *
from utils.utils import text, get_dataset, collate_fn, read_examples, to_list, clean_sentence


accelerator = Accelerator()
if not os.path.exists(save_root) and accelerator.is_local_main_process:
    os.mkdir(save_root)
accelerator.wait_for_everyone()
if not os.path.exists(cache_root) and accelerator.is_local_main_process:
    os.mkdir(cache_root)
accelerator.wait_for_everyone()


GenerationModel = importlib.import_module('models.' + args.model_file).GenerationModel


warnings.filterwarnings("ignore")
device = accelerator.device
train_path = os.path.join(args.data_path, args.dataset, "train_part.json")
eval_path = os.path.join(args.data_path, args.dataset, "valid.json")
test_path = os.path.join(args.data_path, args.dataset, "test.json")


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
    if training:
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'decoder_attention_mask': batch['decoder_attention_mask'],
                    'labels': batch['labels']
                    }
    else:
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'training': False
                    }
    if args.indicator:
        inputs.update({'indicator_ids': batch['indicator_ids']})
    
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
    logging_step = len(train_loader) if args.logging_times == -1 else t_total // args.logging_times
    steps = 0

    # eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
    # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            inputs = prepare_inputs(batch)
            outputs = model(**inputs)
            loss = outputs.loss
            print_loss = loss.item()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()

            scheduler.step()
            model.zero_grad()
            steps += 1

            pbar.set_description("Loss:%.3f,CL:%.3f" %(print_loss, print_loss))
            if steps % logging_step == 0 or steps == t_total:
                accelerator.print("\nEpoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name=f'eval_result-{epoch}.json')
                if accelerator.is_local_main_process and eval_result['ROUGE_L'] < 0.1:
                    accelerator.print("Become untrainable! Training canceled!!!")
                    return
                if accelerator.is_local_main_process and cur_larger(eval_result, best_result):
                    best_result = eval_result
                evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name=f'test_result-{epoch}.json')
                torch.cuda.empty_cache()


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None, cal_metrics=True):
    def _cal_metrics(golden_dict, pred_dict):
        golden_dict = {qid: golden_dict[qid] for qid in pred_dict.keys()}
        evaluator = COCOEvalCap(text(list(golden_dict.values())), text(list(pred_dict.values())))
        evaluator.evaluate()
        result_dict = {metric: score*100 for metric, score in evaluator.eval.items()}
        return result_dict

    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    accelerator.wait_for_everyone()

    model.eval()
    unwrapped_model = accelerator.unwrap_model(deepcopy(model))
    with torch.no_grad():
        pred_dict = {}
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            prefix = batch['qid'][0].split('-')[0]
            inputs = prepare_inputs(batch, training=False)
            summary_ids_list = unwrapped_model(**inputs)
            summary_ids_list = accelerator.pad_across_processes(summary_ids_list, dim=1, pad_index=1)
            qid_tensors = torch.tensor([int(x.split('-')[1]) for x in batch['qid']]).to(device)
            all_summary_ids, all_qid_tensors = accelerator.gather((summary_ids_list, qid_tensors))
            all_qids = [prefix+'-'+str(x) for x in to_list(all_qid_tensors)]
            for summary_ids, qid in zip(all_summary_ids, all_qids):
                decoded_summary = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pred_dict[qid] = clean_sentence(decoded_summary)

    if accelerator.is_local_main_process and cal_metrics:
        examples = read_examples(test_path if is_test else eval_path, training=False, accelerator=accelerator)
        golden_dict = {exp.qid: exp.response_info['response'] for exp in examples}
        result_dict = _cal_metrics(golden_dict, pred_dict)
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


def training(model, tokenizer, decode_tokenizer):
    train_dataset = get_dataset(train_path, tokenizer, decode_tokenizer=decode_tokenizer, training=True, accelerator=accelerator)
    eval_dataset = get_dataset(eval_path, tokenizer, decode_tokenizer=decode_tokenizer, training=False, accelerator=accelerator)
    test_dataset = get_dataset(test_path, tokenizer, decode_tokenizer=decode_tokenizer, training=False, accelerator=accelerator)
    num_workers = CPU_COUNT if args.full_data else 0
    # num_workers = 16
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers)

    if 'None' not in args.pretrain_path:
        accelerator.print("Loading pre-trained model from {}...".format(args.pretrain_path))
        model_saved = torch.load(args.pretrain_path)
        real_model_saved = {}
        for k in model_saved.keys():
            if args.encoder_model_type in k:
                real_model_saved[k.replace(args.encoder_model_type+'.', '')] = model_saved[k]
        if args.decoder_model_type == 'bart':
            missing_keys, unexpected_keys = model.base_encoder.load_state_dict(real_model_saved, strict=False)
        else:
            missing_keys, unexpected_keys = model.model.encoder.load_state_dict(real_model_saved, strict=False)
        accelerator.print("Missing keys: {}\nUnexpected_keys: {}\n".format(missing_keys, unexpected_keys))
        if args.indicator:
            indicator_weights = {"weight": model_saved['indicator_embs.weight']}
            model.indicator_embs.load_state_dict(indicator_weights)

    model = model.to(device)
    train(model, train_dataloader, eval_dataloader, test_dataloader, decode_tokenizer)


def inference(model, tokenizer, decode_tokenizer):
    model_path = "ubuntu_saves/bart_addrmode0_embindicatorlr1e-05_lgsave_384_1919810/"
    # model_path = "ubuntu_saves/bart_addrmode0_embindicatorlr4e-05_save_384_1919810/"
    eval_path = os.path.join(args.data_path, args.dataset, "case.json")
    # args = torch.load(model_path + 'args.pth')

    eval_dataset = get_dataset(eval_path, tokenizer, decode_tokenizer=decode_tokenizer, training=False)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    # test_dataset = get_dataset(test_path, tokenizer, decode_tokenizer=decode_tokenizer, training=False)
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    # if hasattr(model.model, 'load_mha_params'):
    #     accelerator.print("Loading multi-head attention parameters from pretrained model...")
    #     model.model.load_mha_params()
    
    model = model.to(device)

    model_saved = torch.load(model_path + "best_model.pth")
    model.load_state_dict(model_saved)
    # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True,\
    #         file_name="result_retest")
    evaluate(model, eval_dataloader, decode_tokenizer, file_name="case_results_valid.json", cal_metrics=False)


def main():
    set_seed()

    def _get_cache_dir(model_name):
        return os.path.join(args.cache_path,\
             model_name+'{}_cache'.format('_large' if 'large' in model_name else ''))

    if accelerator.is_local_main_process:
        print("Main process loading tokenizer and model...")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        decode_tokenizer_name = args.decoder_model_name if args.decoder_model_type=='bart' else args.model_name
        decode_tokenizer = AutoTokenizer.from_pretrained(decode_tokenizer_name, cache_dir=_get_cache_dir(\
             args.decoder_model_type if args.decoder_model_type=='bart' else args.encoder_model_type))
        config_encoder = AutoConfig.from_pretrained(args.model_name)
        config_decoder = AutoConfig.from_pretrained(args.decoder_model_name, cache_dir=_get_cache_dir(args.decoder_model_type))

        if args.encoder_model_type == 'bert':
            encoder = AutoModel.from_pretrained(args.model_name, config=config_encoder, add_pooling_layer=False)
        else:
            encoder = AutoModel.from_pretrained(args.model_name, config=config_encoder)
        if args.decoder_model_type == 'bart':
            model = GenerationModel.from_pretrained(args.decoder_model_name, config=config_decoder,\
                 cache_dir=_get_cache_dir(args.decoder_model_type))
            model.__setattr__("base_encoder", encoder)
        else:
            config_decoder.is_decoder = True
            config_decoder.add_cross_attention = True
            decoder = AutoModelForCausalLM.from_pretrained(args.decoder_model_name, config=config_decoder,\
                 cache_dir=_get_cache_dir(args.decoder_model_type))
            model = GenerationModel(encoder=encoder, decoder=decoder, decoder_start_token_id=tokenizer.cls_token_id,\
                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id)

    accelerator.wait_for_everyone() # main process downloads and caches the pre-trained models, other processes wait


    if not accelerator.is_local_main_process: # other processes load from cache
        print("Follower process {} loading tokenizer and model...\n".format(accelerator.local_process_index))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        decode_tokenizer_name = args.decoder_model_name if args.decoder_model_type=='bart' else args.model_name
        decode_tokenizer = AutoTokenizer.from_pretrained(decode_tokenizer_name, cache_dir=_get_cache_dir(\
             args.decoder_model_type if args.decoder_model_type=='bart' else args.encoder_model_type))
        config_encoder = AutoConfig.from_pretrained(args.model_name)
        config_decoder = AutoConfig.from_pretrained(args.decoder_model_name, cache_dir=_get_cache_dir(args.decoder_model_type))

        if args.encoder_model_type == 'bert':
            encoder = AutoModel.from_pretrained(args.model_name, config=config_encoder, add_pooling_layer=False)
        else:
            encoder = AutoModel.from_pretrained(args.model_name, config=config_encoder)
        if args.decoder_model_type == 'bart':
            model = GenerationModel.from_pretrained(args.decoder_model_name, config=config_decoder,\
                 cache_dir=_get_cache_dir(args.decoder_model_type))
            model.__setattr__("base_encoder", encoder)
        else:
            config_decoder.is_decoder = True
            config_decoder.add_cross_attention = True
            decoder = AutoModelForCausalLM.from_pretrained(args.decoder_model_name, config=config_decoder,\
                 cache_dir=_get_cache_dir(args.decoder_model_type))
            model = GenerationModel(encoder=encoder, decoder=decoder, decoder_start_token_id=tokenizer.cls_token_id,\
                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id)

    tokenizer.truncation_side = 'left'
    accelerator.wait_for_everyone()

    # tmp(train_path, file_name="train_addr_preds.json")
    training(model, tokenizer, decode_tokenizer)
    # inference(model, tokenizer)


if __name__ == "__main__":
    main()
