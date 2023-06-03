import importlib
import json
import os
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (AdamW, BertConfig, BertTokenizerFast,
                          DebertaV2Config, DebertaV2Tokenizer, ElectraConfig,
                          ElectraTokenizerFast,
                          get_linear_schedule_with_warmup)

from utils.utils_full import *
time.sleep(5)

accelerator = Accelerator()
accelerator.wait_for_everyone()
if accelerator.is_local_main_process and not os.path.exists(args.save_root):
    os.mkdir(args.save_root)
accelerator.wait_for_everyone()
if accelerator.is_local_main_process and not os.path.exists(args.cache_root):
    os.mkdir(args.cache_root)
accelerator.wait_for_everyone()

other_training_states = otherTrainingStates()


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast),
    'deberta': (DebertaV2Config, DebertaV2Tokenizer)
}
KEEP_LAYER_KEYS = ["encoder.layer.{}.".format(i) for i in range(args.keep_layers)] +\
     ['cls.', 'lm_head.', 'link_classifier.', 'response_classifier']
BIGGER_LR_KEYS = ['cls.', 'lm_head.', 'indicator_embs.', '.classifier.', '.response_classifier']
BIG_RATIO = 2


warnings.filterwarnings("ignore")
device = accelerator.device
eval_path = os.path.join(args.data_path, args.dataset, "valid.json")
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


def prepare_inputs(batch, training=True, estep=False, tau=0.1):
    inputs = {'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'token_type_ids': batch['token_type_ids'],
                'sep_poses': batch['sep_poses']
                }
    if training:
        inputs.update({'labels': batch['labels']})
    if args.indicator:
        inputs.update({'indicator_ids': batch['indicator_ids']})
    if estep:
        inputs.update({'e_step': True})
    if args.mlm_weight > 0.0 and training:
        inputs.update({'mlm_dict': batch['mlm_dict']})
    if args.attn_weight > 0.0 and training:
        inputs.update({'attention_guidance': batch['attention_guidance']})
    if args.link_weight > 0.0:
        inputs.update({'link_labels': batch['link_labels'], 'link_masks': batch['link_masks'], 'tau': tau})
    return inputs


def train(total_trunk_number, eval_example_lines, eval_record_lines, tokenizer):
    accelerator.print("Training arguments:")
    accelerator.print(args)
    accelerator.print("\n"+"="*128+"\n", "Start training, total trunk number: {}!!!".format(total_trunk_number), "\n"+"="*128+"\n")

    if args.resume:
        # tell if is continuous training
        checkpoint_path = os.path.join(args.save_path, "last_states")
        other_checkpoint_path = os.path.join(checkpoint_path, "others.json")
        best_unwrapped_save_path = os.path.join(checkpoint_path, "last_best_unwrapped_model.pth")
        trunk_best_model_path = os.path.join(checkpoint_path, "trunk_best_unwrapped_model.pth")
        FROM_CHECKPOINT = os.path.exists(checkpoint_path)
        accelerator.wait_for_everyone()
        if FROM_CHECKPOINT: # resuming starting trunk index, update addr_preds.json ans so on
            other_training_states.load_state(other_checkpoint_path)
            training_state_dict = other_training_states.state_dict()
            trunk_idx, trunk_best_score, trunk_confidence_method, best_result, best_addr_acc, best_epoch, global_best_addr_acc,\
                 no_improvement_turns, first_eval_addr_acc, pred_addrs, confidence_method = training_state_dict["trunk_idx"],\
                 training_state_dict["trunk_best_score"], training_state_dict["trunk_confidence_method"],\
                 training_state_dict["best_result"], training_state_dict["best_addr_acc"], training_state_dict["best_epoch"],\
                 training_state_dict["global_best_addr_acc"], training_state_dict["no_improvement_turns"], training_state_dict['first_eval_addr_acc'],\
                 training_state_dict['pred_addrs'], training_state_dict['confidence_method']
            if accelerator.is_local_main_process: # assert pred_addrs is not None
                with open(os.path.join(args.save_path, "addr_preds.json"), "w", encoding='utf-8') as f:
                    json.dump(pred_addrs, f, indent=2)
            if trunk_idx >= args.stage_two_trunk:
                args.model_file = args.model_file.replace("pooler", "link_fast")

    if not args.resume or not FROM_CHECKPOINT: # initialization
        trunk_idx = 0
        confidence_method = "rank"
        trunk_best_score, trunk_confidence_method = 0.0, "rank"
        best_result = {metric: 0.0 for metric in METRICS}
        best_addr_acc, best_epoch, global_best_addr_acc, no_improvement_turns, first_eval_addr_acc = 0, 0, 0, 0, None
    accelerator.wait_for_everyone()
    
    SHOULD_RESTART, model, best_unwrapped_model, trunk_best_unwrapped_model, train_dataset_estep = False, None, None, None, None
    accelerator.wait_for_everyone()

    while trunk_idx < total_trunk_number:
        accelerator.print("\n"+"="*128+"\n", "Start to train trunk {}/{}!!!".format(trunk_idx, total_trunk_number), "\n"+"="*128+"\n")
        args.reinit = max(args.min_mstep_rounds, args.reinit - trunk_idx)
        args.init_rounds = max(args.min_mstep_rounds, args.init_rounds - trunk_idx)
        args.trunk_em_rounds = max(2, args.trunk_em_rounds - trunk_idx)
        accelerator.print("\n"+"="*128+"\n", "Current init_rounds: {}, re-init rounds {}".format(args.init_rounds, args.reinit), "\n"+"="*128+"\n")

        if model is None:
            accelerator.print("Loading config and model weights...")
            config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
            MultipleChoiceModel = importlib.import_module('models.' + args.model_file).MultipleChoiceModel
            model = MultipleChoiceModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
            accelerator.print("Loading is done!")
            model.train()
            model.zero_grad()
            accelerator.wait_for_everyone()

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(bgk in n for bgk in BIGGER_LR_KEYS) and p.requires_grad]},
            {'params': [p for n, p in model.named_parameters() if any(bgk in n for bgk in BIGGER_LR_KEYS) and p.requires_grad], "lr": args.learning_rate * max(1, BIG_RATIO-trunk_idx)}
        ]
        all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        # dataset for E-step
        if train_dataset_estep is None:
            train_dataset_estep = get_dataset(tokenizer, "train", trunk_idx=trunk_idx, confidence_method=confidence_method,\
                training=True, e_step=True, accelerator=accelerator)
            train_sampler_estep = SequentialSampler(train_dataset_estep)
            train_loader_estep = DataLoader(train_dataset_estep, sampler=train_sampler_estep,\
                batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.cpu_count)
            train_loader_estep = accelerator.prepare(train_loader_estep)
        # dataset for M-step
        train_dataset = get_dataset(tokenizer, "train",  trunk_idx=trunk_idx, confidence_method=trunk_confidence_method,\
            training=True, e_step=False, accelerator=accelerator) # TODO maybe change back
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,\
            collate_fn=collate_fn, num_workers=args.cpu_count)
        # dataset for evaluation
        eval_dataset = get_dataset(tokenizer, 'valid', record_lines=eval_record_lines, example_lines=eval_example_lines, training=False, accelerator=accelerator)
        eval_dataset_estep = get_dataset(tokenizer, 'valid', record_lines=eval_record_lines, example_lines=eval_example_lines, training=True, e_step=True, accelerator=accelerator)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size,\
            collate_fn=collate_fn, num_workers=args.cpu_count)
        eval_sampler_estep = SequentialSampler(eval_dataset_estep)
        eval_dataloader_estep = DataLoader(eval_dataset_estep, sampler=eval_sampler_estep,\
            batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.cpu_count)
        
        t_total = len(train_loader) * args.init_rounds
        num_warmup_steps = int(t_total * args.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

        model, all_optimizer, train_loader, eval_dataloader, eval_dataloader_estep, scheduler = accelerator.prepare(
            model, all_optimizer, train_loader, eval_dataloader, eval_dataloader_estep, scheduler)
        
        if not args.resume or not FROM_CHECKPOINT:
            epoch = 0

        if args.resume:
            if not FROM_CHECKPOINT:
                with open(os.path.join(args.save_path, "addr_preds.json"), "r", encoding='utf-8') as f:
                    pred_addrs = json.load(f) # read out the predicted addressees of e-step initialization for saving if the training is not from checkpoint
            accelerator.wait_for_everyone()
            # continue training from checkpoint
            if FROM_CHECKPOINT: # load the training states
                accelerator.print("\n"+"="*128 + "\nResuming training states from checkpoint!!!\n" + "="*128+"\n")
                accelerator.wait_for_everyone()
                accelerator.load_state(checkpoint_path)
                # re-loading best model for e-step
                if os.path.exists(best_unwrapped_save_path):
                    best_unwrapped_model = accelerator.unwrap_model(deepcopy(model)).to("cpu")
                    best_unwrapped_model.load_state_dict(torch.load(best_unwrapped_save_path))
                    best_unwrapped_model = best_unwrapped_model.to(accelerator.device)
                if os.path.exists(trunk_best_model_path):
                    trunk_best_unwrapped_model = accelerator.unwrap_model(deepcopy(model)).to("cpu")
                    trunk_best_unwrapped_model.load_state_dict(torch.load(trunk_best_model_path))
                    trunk_best_unwrapped_model = trunk_best_unwrapped_model.to(accelerator.device)
                accelerator.print("\n"+"="*128 + "\nResuming process done!!!\n" + "="*128+"\n")
                accelerator.wait_for_everyone()
                epoch = training_state_dict['epoch']
                FROM_CHECKPOINT = False # next trunk don't resume

        while epoch < args.reinit * args.trunk_em_rounds + args.init_rounds:
            if args.resume:
                # save the current state of model, optimizer, scheduler, trunk_idx, and addr_preds
                accelerator.print("\n"+"="*128 + "\nSaving training states to checkpoint!!!\n" + "="*128+"\n")
                accelerator.wait_for_everyone()
                accelerator.save_state(checkpoint_path) # model, optimizer, scheduler, random states
                with open(os.path.join(args.save_path, "addr_preds.json"), "r", encoding='utf-8') as f:
                    pred_addrs = json.load(f)
                training_state_dict = {"trunk_idx": trunk_idx, "trunk_best_score": trunk_best_score, "trunk_confidence_method": trunk_confidence_method,\
                    "epoch": epoch, "best_result": best_result, "best_addr_acc": best_addr_acc, "best_epoch": best_epoch,\
                    "global_best_addr_acc": global_best_addr_acc, "no_improvement_turns": no_improvement_turns,\
                    "first_eval_addr_acc": first_eval_addr_acc,"confidence_method": confidence_method, "pred_addrs": pred_addrs}
                other_training_states.update(training_state_dict)
                if accelerator.is_local_main_process:
                    other_training_states.save_state_dict(other_checkpoint_path)
                accelerator.wait_for_everyone()
                if best_unwrapped_model is not None:
                    saved_best_unwrapped_model = best_unwrapped_model.to("cpu")
                    if accelerator.is_local_main_process:
                        torch.save(saved_best_unwrapped_model.state_dict(), best_unwrapped_save_path)
                    accelerator.wait_for_everyone()
                    best_unwrapped_model.to(accelerator.device)
                if trunk_best_unwrapped_model is not None:
                    saved_trunk_best_unwrapped_model = trunk_best_unwrapped_model.to("cpu")
                    if accelerator.is_local_main_process:
                        torch.save(saved_trunk_best_unwrapped_model.state_dict(), trunk_best_model_path)
                    accelerator.wait_for_everyone()
                    trunk_best_unwrapped_model.to(accelerator.device)
                accelerator.print("\n"+"="*128 + "\nSaving is done!!!\n" + "="*128+"\n")
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                gc.collect()

            # start training
            re_init_round = (epoch-args.init_rounds) // max(args.reinit, 1) + 1 if epoch-args.init_rounds>=0 else 0
            accelerator.print("\n"+"="*128+"\n", "Start to train trunk {}, re-init round {}, epoch {}!!!".format(trunk_idx, re_init_round, epoch), "\n"+"="*128+"\n")
            # maximization step
            accelerator.wait_for_everyone()
            accelerator.print("Maximization step in trunk {}, reinit {}, epoch {}...".format(trunk_idx, re_init_round, epoch))
            m_step(model, train_loader, all_optimizer, scheduler, trunk_idx=trunk_idx)
            accelerator.wait_for_everyone()
            # evaluate addressee prediction on validation set
            eval_acc, last_eval_pred_adr_list, cur_confidence_method, confidence_eval_acc = e_step_eval(model, eval_dataloader_estep, eval_path, file_name="eval_addr_preds.json", trunk_idx=trunk_idx)
            accelerator.print("Trunk {}, Reinit {}, Epoch {}, Addressee prediction of E-step on validation set: {}%".format(trunk_idx, re_init_round, epoch, eval_acc))

            # see if the training is stuck or become untrainable (when using ELECTRA-large, it may happen)
            if first_eval_addr_acc is None: first_eval_addr_acc = eval_acc
            if eval_acc < first_eval_addr_acc - (40.0 if args.model_type=='bert' else 10.0):
                accelerator.print("\n"+"="*128 + "\nBecome untrainable in trunk {}, re-init round {}, epoch {}, Re-start!!!\n".format(trunk_idx, re_init_round, epoch) + "="*128+"\n")
                SHOULD_RESTART = True
            accelerator.wait_for_everyone()
        
            if not SHOULD_RESTART:
                # update the addressee list of eval dataloader
                eval_dataloader.dataset.adr_list = last_eval_pred_adr_list
                accelerator.wait_for_everyone()
                # evaluate response selection on validation set
                eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json', trunk_idx=trunk_idx)
                accelerator.print("\nTrunk {}, Reinit {}, Epoch {}, validation Result: {}".format(trunk_idx, re_init_round, epoch, eval_result))
                if cur_larger(eval_result, best_result):
                    best_result = eval_result
                accelerator.wait_for_everyone()
                # saving the best model with the highest addr_acc+r1
                if eval_acc > best_addr_acc:
                    accelerator.print("The current addressee prediction accuracy in trunk {} reinit {}, epoch {} is the best in this re-init round, saving the model...".format(\
                         trunk_idx, re_init_round, epoch))
                    best_addr_acc = eval_acc
                    best_epoch = epoch
                    confidence_method = cur_confidence_method
                    model.eval()
                    del best_unwrapped_model
                    torch.cuda.empty_cache()
                    gc.collect()
                    best_unwrapped_model = accelerator.unwrap_model(deepcopy(model))
                    model.train()
                    # save checkpoints
                    if accelerator.is_local_main_process:
                        save_path = os.path.join(args.save_path, "best-trunk-{}-reinit-{}.pth".format(trunk_idx, re_init_round))
                        args_save_path = os.path.join(args.save_path, "args.pth")
                        model.eval()
                        unwrapped_model = accelerator.unwrap_model(deepcopy(model)).to("cpu")
                        accelerator.save(unwrapped_model.state_dict(), save_path)
                        accelerator.save(args, args_save_path)
                        model.train()
                        del unwrapped_model
                        torch.cuda.empty_cache()
                        gc.collect()
                accelerator.wait_for_everyone()
                cur_score = confidence_eval_acc + round(0.5*eval_result["r@1"]*100, 3)
                if cur_score > trunk_best_score:
                    accelerator.print("The current score (confidence_addr + 0.5*R@1) in trunk {} reinit {}, epoch {} is the best in this TRUNK, saving the model...".format(\
                         trunk_idx, re_init_round, epoch))
                    trunk_best_score = cur_score
                    trunk_confidence_method = cur_confidence_method
                    model.eval()
                    del trunk_best_unwrapped_model
                    torch.cuda.empty_cache()
                    gc.collect()
                    trunk_best_unwrapped_model = accelerator.unwrap_model(deepcopy(model))
                    model.train()
                accelerator.wait_for_everyone()

            # expectation step OR re-start
            if epoch >= args.init_rounds-1 and (epoch+1-args.init_rounds) % args.reinit == 0 or SHOULD_RESTART: # epoch init_rounds should reinit
                for_next_trunk = (epoch + 1) == args.reinit * args.trunk_em_rounds + args.init_rounds
                if not SHOULD_RESTART: # E-step, if we don't get here by SHOULD_RESTART
                    # detect early stop
                    if best_addr_acc > global_best_addr_acc:
                        no_improvement_turns = 0
                        global_best_addr_acc = best_addr_acc
                    else:
                        no_improvement_turns += 1
                    if no_improvement_turns >= args.patience:
                        accelerator.print("Running out of patience {}/{}, early stop!".format(no_improvement_turns, args.patience))
                        exit(0)
                    best_addr_acc = 0.0 # reset the best addressee prediction accuracy in this re-init round
                    # expectation step
                    if not for_next_trunk: # E-step for the next M-step in this trunk
                        accelerator.print("\n"+"="*128+"\n",\
                             "Expectation step in trunk {}, reinit {}, epoch {}, best epoch is {}, confidence method is {}...".format(\
                                 trunk_idx, re_init_round, epoch, best_epoch, confidence_method), "\n"+"="*128+"\n")
                        e_step(best_unwrapped_model, train_loader_estep, "-{}-{}".format(trunk_idx, re_init_round), "addr_preds{}.json", trunk_idx=trunk_idx)
                        accelerator.print("\n"+"="*128+"\n", "Expectation step done!!!", "\n"+"="*128+"\n")
                        accelerator.wait_for_everyone()
                
                # re-initialize accelerator and release memory etc.
                accelerator.print("Re-initializing dataloader, model, optimizer and scheduler...")
                accelerator.clear() # THIS IS REALLY IMPORTANT WHEN RE-INITIALIZATION !!!

                if not for_next_trunk:
                    accelerator.print("Re-initialize dataloaders for the current trunk...")
                    # should reset dataloaders that have been prepared after accelerator.clear() for E-step/normal training/evaluation!!!
                    eval_dataset = get_dataset(tokenizer, 'valid', record_lines=eval_record_lines, example_lines=eval_example_lines, training=False, accelerator=accelerator)
                    eval_sampler = SequentialSampler(eval_dataset)
                    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size,\
                        collate_fn=collate_fn, num_workers=args.cpu_count)
                    
                    eval_dataset_estep = get_dataset(tokenizer, 'valid', record_lines=eval_record_lines, example_lines=eval_example_lines, training=True, e_step=True, accelerator=accelerator)
                    eval_sampler_estep = SequentialSampler(eval_dataset_estep)
                    eval_dataloader_estep = DataLoader(eval_dataset_estep, sampler=eval_sampler_estep,\
                        batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.cpu_count)
                    
                    train_dataset_estep = get_dataset(tokenizer, "train", trunk_idx=trunk_idx, confidence_method=confidence_method,\
                        training=True, e_step=True, accelerator=accelerator)
                    train_sampler_estep = SequentialSampler(train_dataset_estep)
                    train_loader_estep = DataLoader(train_dataset_estep, sampler=train_sampler_estep,\
                        batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.cpu_count)
                    train_loader_estep, eval_dataloader, eval_dataloader_estep = accelerator.prepare(
                            train_loader_estep, eval_dataloader, eval_dataloader_estep)
                    accelerator.wait_for_everyone()

                # re-initialize model
                accelerator.print("Re-initialize model and loading preserved weights...")
                if trunk_idx == args.stage_two_trunk-1 and for_next_trunk and "link_fast" not in args.model_file:
                    args.model_file = args.model_file.replace("pooler", "link_fast")
                config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
                MultipleChoiceModel = importlib.import_module('models.' + args.model_file).MultipleChoiceModel
                model = MultipleChoiceModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
                # re-load previous weight of shallow layers
                if args.keep_layers > 0 or args.mlm_weight > 0.0:
                    accelerator.print("\n"+"="*128+"\n", "Loading weights from the last best checkpoint...", "\n"+"="*128+"\n")
                    last_state_dict = deepcopy(trunk_best_unwrapped_model).to("cpu").state_dict() if for_next_trunk else deepcopy(best_unwrapped_model).to("cpu").state_dict() # TODO maybe change back
                    last_state_keys = list(last_state_dict.keys())
                    for k in last_state_keys:
                        should_delete = True
                        for keep_k in KEEP_LAYER_KEYS:
                            if keep_k in k:
                                should_delete = False
                                break
                        if should_delete: del last_state_dict[k]
                    missing_keys, unexpected_keys = model.load_state_dict(last_state_dict, strict=False)
                    accelerator.print("Missing keys: {}\nUnexpected_keys: {}\n".format(missing_keys, unexpected_keys))
                    accelerator.print("\n"+"="*128+"\n", "Loading is done!!!", "\n"+"="*128+"\n")
                    accelerator.wait_for_everyone()
                model.train()
                model.zero_grad()

                if not for_next_trunk:
                    accelerator.print("Re-initialize train_loader, optimizer, scheduler for the current trunk...")
                    # re-initialize optimizer
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in model.named_parameters() if not any(bgk in n for bgk in BIGGER_LR_KEYS) and p.requires_grad]},
                        {'params': [p for n, p in model.named_parameters() if any(bgk in n for bgk in BIGGER_LR_KEYS) and p.requires_grad], "lr": args.learning_rate * max(1, BIG_RATIO-trunk_idx)}
                    ]
                    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
                    # re-initialize train_loader, should prepare it AFTER setting t_total
                    train_dataset = get_dataset(tokenizer, "train",  trunk_idx=trunk_idx, confidence_method=confidence_method,\
                        training=True, e_step=False, accelerator=accelerator)
                    train_sampler = RandomSampler(train_dataset)
                    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,\
                        collate_fn=collate_fn, num_workers=args.cpu_count)
                    accelerator.wait_for_everyone()
                    # re-initialize scheduler
                    t_total = len(train_loader) * args.reinit
                    num_warmup_steps = int(t_total * args.warmup_proportion)
                    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
                    # preparing
                    model, all_optimizer, train_loader, scheduler = accelerator.prepare(model, all_optimizer, train_loader, scheduler) # the CUDA-OOM problem happens here
                    accelerator.print("Re-initialization done.")
                elif trunk_idx+1 < total_trunk_number: # E-step to initialize addressees for the next trunk
                    # dataset for E-step
                    train_dataset_estep = get_dataset(tokenizer, "train", trunk_idx=trunk_idx+1, confidence_method=confidence_method,\
                        training=True, e_step=True, accelerator=accelerator)
                    train_sampler_estep = SequentialSampler(train_dataset_estep)
                    train_loader_estep = DataLoader(train_dataset_estep, sampler=train_sampler_estep,\
                        batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.cpu_count)
                    train_loader_estep = accelerator.prepare(train_loader_estep)
                    accelerator.print("\n"+"="*128+"\n", "Expectation step for the next trunk, confidence method {}...".format(trunk_confidence_method), "\n"+"="*128+"\n")
                    e_step(trunk_best_unwrapped_model, train_loader_estep, "-{}-init".format(trunk_idx+1), "addr_preds{}.json", trunk_idx=trunk_idx) # TODO maybe change back
                    accelerator.print("\n"+"="*128+"\n", "Expectation step done!!!", "\n"+"="*128+"\n")

                if SHOULD_RESTART:
                    epoch -= 1 # this epoch is invalid
                    SHOULD_RESTART = False
                torch.cuda.empty_cache()
                gc.collect()
                accelerator.wait_for_everyone()
            # add the epoch index for the next epoch
            epoch += 1
        # add the trunk_idx for the next trunk, reset some variables
        trunk_best_score = 0.0
        trunk_idx += 1


def e_step(unwrapped_model, eval_loader, save_postfix, file_name=None, cache_path=None, trunk_idx=0):
    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    accelerator.wait_for_everyone()

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "r", encoding='utf-8') as f:
            pred_addrs = json.load(f)
        if accelerator.is_local_main_process and file_name is not None:
            with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
                json.dump(pred_addrs, f, indent=2)
        accelerator.wait_for_everyone()
        return

    unwrapped_model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        prefix = eval_loader.dataset.prefix
        log_prob_dict = {}

        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False, estep=True, tau=1/max(0.1, 1/(trunk_idx-args.stage_two_trunk+0.1)))
            log_probs = unwrapped_model(**inputs)[0]
            accelerator.wait_for_everyone()
            all_log_probs, all_qid_tensors = accelerator.gather((log_probs, batch['qid_tensor'])) # (bsz)
            all_log_probs = to_list(all_log_probs)
            all_qids = to_list(all_qid_tensors)
            for index, log_prob in zip(all_qids, all_log_probs):
                context_idx, offset, addr = eval_loader.dataset.examples[index].split('\t') # this is the true addr (before truncation)
                qqid = prefix + '-' + context_idx + '-' + offset + '-' + addr
                log_prob_dict[qqid] = log_prob

    if accelerator.is_local_main_process:
        sample_log_prob_dict = defaultdict(list)
        for qqid, log_prob in log_prob_dict.items():
            qid = qqid[:qqid.rfind('-')]
            addr = int(qqid[qqid.rfind('-')+1:])
            sample_log_prob_dict[qid].append((addr, log_prob)) # addr may not start from 0, hence we should store it
        
        pred_addrs = {}
        for qid, sample_probs in sample_log_prob_dict.items():
            sorted_probs = sorted(sample_probs, key=lambda x: x[0])
            addrs = [tp[0] for tp in sorted_probs]
            log_probs = [tp[1] for tp in sorted_probs]
            pred_addr = addrs[log_probs.index(max(log_probs))]
            pred_addrs[qid] = {'pred': pred_addr, 'log_probs': log_probs}

        if file_name is not None:
            with open(os.path.join(args.save_path, file_name.format('')), "w", encoding='utf-8') as f:
                json.dump(pred_addrs, f, indent=2)
            if cache_path is None:
                with open(os.path.join(args.save_path, file_name.format(save_postfix)), "w", encoding='utf-8') as f:
                    json.dump(pred_addrs, f, indent=2)

        if cache_path is not None:
            with open(cache_path, "w", encoding='utf-8') as f:
                json.dump(pred_addrs, f, indent=2)

    accelerator.wait_for_everyone()


def e_step_eval(model, eval_loader, file_path, file_name, trunk_idx=0):
    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    accelerator.wait_for_everyone()

    model.eval()
    prefix = eval_loader.dataset.prefix
    unwrapped_model = accelerator.unwrap_model(deepcopy(model))
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        log_prob_dict = {}

        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False, estep=True, tau=1/max(0.1, 1/(trunk_idx-args.stage_two_trunk+0.1)))
            log_probs = unwrapped_model(**inputs)[0]
            accelerator.wait_for_everyone()
            all_log_probs, all_qid_tensors = accelerator.gather((log_probs, batch['qid_tensor'])) # (bsz)
            all_log_probs = to_list(all_log_probs)
            all_qids = to_list(all_qid_tensors)
            for index, log_prob in zip(all_qids, all_log_probs):
                context_idx, offset, addr = eval_loader.dataset.examples[index].split('\t') # this is the true addr (before truncation)
                qqid = prefix + '-' + context_idx + '-' + offset + '-' + addr
                log_prob_dict[qqid] = log_prob

    sample_log_prob_dict = defaultdict(list)
    for qqid, log_prob in log_prob_dict.items():
        qid = qqid[:qqid.rfind('-')]
        addr = int(qqid[qqid.rfind('-')+1:])
        sample_log_prob_dict[qid].append((addr, log_prob)) # addr may not start from 0, hence we should store it
    
    addr_preds = {}
    for qid, sample_probs in sample_log_prob_dict.items():
        sorted_probs = sorted(sample_probs, key=lambda x: x[0])
        addrs = [tp[0] for tp in sorted_probs]
        log_probs = [tp[1] for tp in sorted_probs]
        pred_addr = addrs[log_probs.index(max(log_probs))]
        addr_preds[qid] = {'pred': pred_addr, 'log_probs': log_probs}

    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        if args.small:
            lines = lines[:args.small if 'train' in file_path else args.small//10]
        if "trunks_" in file_path:
            lines = lines[:10000]

    total, correct, adr_list = 0, 0, []
    for idx, line in enumerate(tqdm(lines, ncols=100, disable=not accelerator.is_local_main_process)):
        record = json.loads(line.strip())
        qid = prefix + '-' + str(idx)
        record['ctx_adr'] = record['ctx_adr'] + ([record['ans_adr']] if 'ans_adr' in record else [])
        for offset in range(3, len(record['ctx_adr'])):
            total += 1
            qqid = qid + '-' + str(offset)
            pred_addr = addr_preds[qqid]['pred'] + 1 # the predicted addr index is from 0, +1 to make sure in record, it is from 1
            addr_preds[qqid]['golden'] = record['ctx_adr'][offset] - 1 # record the ground truth
            if record['ctx_adr'][offset] == pred_addr:
                correct += 1
        adr_list.append(record['ctx_adr'])
    eval_acc = round(correct/total*100, 3)
        
    if accelerator.is_local_main_process:
        with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
            json.dump(addr_preds, f, indent=2)
    accelerator.wait_for_everyone()

    pred_adr_list = []
    for qqid, pred_dict in addr_preds.items():
        context_idx, offset = [int(x) for x in qqid.split('-')[1:]]
        if context_idx == len(pred_adr_list):
            pred_adr_list.append(deepcopy(adr_list[context_idx][:offset]))
        pred_adr_list[-1].append(pred_dict['pred'] + 1) # index should be from 1

    # decide confidence method based on evaluation results
    for qqid in addr_preds.keys():
        # min-max scaler
        log_probs = addr_preds[qqid]['log_probs'][:]
        min_log_prob = min(log_probs)
        max_log_prob = max(log_probs)
        denominator = max_log_prob - min_log_prob
        log_probs = [(x-min_log_prob)/denominator for x in log_probs]
        addr_preds[qqid]['minmax_confidence'] = max_log_prob - sorted(log_probs, reverse=True)[1] # max - second

        # avg scaler
        log_probs = addr_preds[qqid]['log_probs'][:]
        min_log_prob = min(log_probs)
        log_probs = [x-min_log_prob for x in log_probs] # make sure all elements > 0
        avg = sum(log_probs) / len(log_probs)
        addr_preds[qqid]['avg_confidence'] = (max(log_probs) - sorted(log_probs, reverse=True)[1]) / avg # (max - second) / avg

    log_dict = {}
    addr_preds = dict(sorted(addr_preds.items(), key=lambda x: x[1]['minmax_confidence'], reverse=True))
    for idx, (qid, _) in enumerate(addr_preds.items()):
        addr_preds[qid]['minmax_rank'] = idx + 1
    if args.force_accuracy > 0.0:
        confidence_rank = int(len(addr_preds) * args.force_accuracy)
        total_last_addr_num, last_addr_correct_num = 0, 0
        correct, last_addr_num = 0, 0
        for qqid in list(addr_preds.keys())[:confidence_rank]:
            offset = int(qqid[qqid.rfind('-')+1:])
            pred_dict = addr_preds[qqid]
            if pred_dict['pred'] == pred_dict['golden']:
                correct += 1
                if pred_dict['golden'] == int(qqid.split('-')[-1]) - 1:
                    last_addr_num += 1
            if pred_dict['golden'] == int(qqid.split('-')[-1]) - 1:
                total_last_addr_num += 1
                if pred_dict['pred'] == pred_dict['golden']: # before this, pred_dict['golden'] is already subtracted by 1, hence don't need pred_dict['pred'] +1 == pred_dict['golden']...
                    last_addr_correct_num += 1
        minmax_acc = round(correct/confidence_rank*100, 3)
        log_dict["minmax_rank"] = {"last_utter_acc": round(last_addr_correct_num/max(total_last_addr_num, 1)*100, 3),\
             "other_acc": round((correct-last_addr_correct_num)/max(1, confidence_rank-total_last_addr_num)*100, 3),\
             "used_last_ratio": round(total_last_addr_num/confidence_rank*100, 3)}
    
    addr_preds = dict(sorted(addr_preds.items(), key=lambda x: x[1]['avg_confidence'], reverse=True))
    for idx, (qid, _) in enumerate(addr_preds.items()):
        addr_preds[qid]['avg_rank'] = idx + 1
    if args.force_accuracy > 0.0:
        confidence_rank = int(len(addr_preds) * args.force_accuracy)
        total_last_addr_num, last_addr_correct_num = 0, 0
        correct, last_addr_num = 0, 0
        for qqid in list(addr_preds.keys())[:confidence_rank]:
            offset = int(qqid[qqid.rfind('-')+1:])
            pred_dict = addr_preds[qqid]
            if pred_dict['pred'] == pred_dict['golden']:
                correct += 1
                if pred_dict['golden'] == int(qqid.split('-')[-1]) - 1:
                    last_addr_num += 1
            if pred_dict['golden'] == int(qqid.split('-')[-1]) - 1:
                total_last_addr_num += 1
                if pred_dict['pred'] == pred_dict['golden']: # before this, pred_dict['golden'] is already subtracted by 1, hence don't need pred_dict['pred'] +1 == pred_dict['golden']...
                    last_addr_correct_num += 1
        avg_acc = round(correct/confidence_rank*100, 3)
        log_dict["avg_rank"] = {"last_utter_acc": round(last_addr_correct_num/max(total_last_addr_num, 1)*100, 3),\
             "other_acc": round((correct-last_addr_correct_num)/max(1, confidence_rank-total_last_addr_num)*100, 3),\
             "used_last_ratio": round(total_last_addr_num/confidence_rank*100, 3)}

        confidence_method = 'minmax_rank' if minmax_acc > avg_acc else 'avg_rank'
        log_prefix = "Training" if "trunks_" in file_path else "Validation"
        accelerator.print("{}: Addressee prediction accuracy: avg:{}% vs. min-max:{}%".format(log_prefix, avg_acc, minmax_acc))
        accelerator.print("{}: Addressee prediction accuracy of used samples is: {}%".format(log_prefix, max(minmax_acc, avg_acc)))
        accelerator.print("{}: Addressee prediction accuracy of last-utterance is: {}%, other is: {}%".format(log_prefix,\
             log_dict[confidence_method]["last_utter_acc"], log_dict[confidence_method]["other_acc"]))
        accelerator.print("{}: Ratio of last utterance in used samples: {}%".format(log_prefix, log_dict[confidence_method]["used_last_ratio"]))

    model.train()
    return eval_acc, pred_adr_list, confidence_method, max(minmax_acc, avg_acc)


def m_step(model, train_loader, all_optimizer, scheduler, trunk_idx=0):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=not accelerator.is_local_main_process)
    for _, batch in pbar:
        inputs = prepare_inputs(batch, tau=1/max(0.1, 1/(trunk_idx-args.stage_two_trunk+0.1)))
        outputs = model(**inputs)
        loss, loss_dict = outputs[0], outputs[1]
        print_loss = loss.item()
        description = "Loss:%.3f," %(print_loss)
        for k, v in loss_dict.items():
            description += f"{k}:{round(v, 3)},"
        description = description[:-1]
        print_loss = loss.item()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        all_optimizer.step()
        scheduler.step()
        model.zero_grad()
        pbar.set_description(description)


def expectation_init(example_lines, record_lines):
    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    accelerator.wait_for_everyone()

    # just regard the previous utterance as the addressee
    if accelerator.is_local_main_process:
        record_start, record_end = int(example_lines[0].split('\t')[0]), int(example_lines[args.trunk_size-1].split('\t')[0]) + 1
        cur_record_lines = record_lines[record_start: record_end]

        pred_addrs = {}
        num = 0
        for idx, line in enumerate(tqdm(cur_record_lines, ncols=100, disable=not accelerator.is_local_main_process)):
            record = json.loads(line.strip())
            qid = 'train-' + str(idx)
            for offset in range(3, len(record['ctx_adr']) + (1 if 'ans_adr' in record else 0)): # +1 for record['ans_adr']
                num += 1
                qqid = qid + '-' + str(offset)
                pred_addrs[qqid] = {}
                pred_addrs[qqid]['pred'] = offset - 1 # index from 0
                pred_addrs[qqid]['log_probs'] = [0.0]*(offset-1) + [1.0]
                pred_addrs[qqid]['rank'] = num
        with open(os.path.join(args.save_path, "addr_preds.json"), "w", encoding='utf-8') as f:
            json.dump(pred_addrs, f, indent=2)

    accelerator.wait_for_everyone()
    accelerator.print("Initialization done!")


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None, cal_metrics=True, trunk_idx=0):
    def _cal_metrics(golden_dict, pred_dict):
        r1, r2, mrr = 0, 0, 0
        for qqid, pred_list in pred_dict.items():
            golden_list = golden_dict[qqid]
            assert golden_list[0] == 1 and sum([max(0, x) for x in golden_list]) == 1
            sorted_preds = [y[0] for y in sorted(enumerate(pred_list), key=lambda x: x[1], reverse=True)]
            ans_position = sorted_preds.index(golden_list.index(1)) + 1 # here golden_list.index(1) == 0
            if ans_position == 1:
                r1 += 1
                r2 += 1
            elif ans_position == 2:
                r2 += 1
            mrr += 1 / ans_position
        r1 /= len(pred_dict)
        r2 /= len(pred_dict)
        mrr /= len(pred_dict)
        return {"r@1": r1, "r@2": r2, "mrr": mrr}

    if accelerator.is_local_main_process and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    accelerator.wait_for_everyone()

    model.eval()
    prefix = eval_loader.dataset.prefix
    unwrapped_model = accelerator.unwrap_model(deepcopy(model))
    with torch.no_grad():
        pred_dict = {}
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False, tau=1/max(0.1, 1/(trunk_idx-args.stage_two_trunk+0.1)))
            pred_probs = unwrapped_model(**inputs)[0]
            accelerator.wait_for_everyone()
            all_pred_probs, all_qid_tensors = accelerator.gather((pred_probs, batch['qid_tensor']))
            all_qids = [prefix+'-'+str(x) for x in to_list(all_qid_tensors)]
            for pred, qid in zip(to_list(all_pred_probs), all_qids):
                pred_dict[qid] = pred

    if cal_metrics:
        answer_path = eval_path
        prefix = 'test' if is_test else 'valid'
        
        example_path_format = answer_path.replace(".json", "{}.tsv")
        example_postfix = "_negamode_{}".format(args.negative)
        if args.small and args.small < 100000:
            example_postfix += '_small_{}'.format(args.small if prefix=='train' else args.small//10)
        example_path = example_path_format.format(example_postfix)

        with open(example_path, "r", encoding='utf-8') as f:
            lines = f.readlines()

        accelerator.wait_for_everyone()
        real_pred_dict, golden_dict = defaultdict(list), defaultdict(list)
        for idx, line in enumerate(lines):
            context_idx, offset, _, _, label = [int(x) for x in line.strip().split('\t')]
            qqid = "{}-{}-{}".format(prefix, context_idx, offset)
            golden_dict[qqid].append(label)
            pred = pred_dict[prefix+'-'+str(idx)]
            real_pred_dict[qqid].append(pred)

        result_dict = _cal_metrics(golden_dict, real_pred_dict)
        if accelerator.is_local_main_process and cur_best_result is not None and cur_larger(result_dict, cur_best_result):
            accelerator.print("model and arguments saved to {}...".format(args.save_path))
            save_path = os.path.join(args.save_path, "best_model.pth")
            args_save_path = os.path.join(args.save_path, "args.pth")
            unwrapped_model = unwrapped_model.to("cpu")
            accelerator.save(unwrapped_model.state_dict(), save_path)
            accelerator.save(args, args_save_path)
        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process and file_name is not None:
        with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
            json.dump(real_pred_dict, f, indent=2)
    accelerator.wait_for_everyone()

    del unwrapped_model
    model.train()

    return result_dict if cal_metrics else None


def evaluate_link_prediction(model, eval_loader):
    def _cal_metrics(link_labels, link_preds, labels):
        cnt_golden, cnt_pred, cnt_cor_bi = 0, 0, 0
        for hypothesis, reference, crm_label in zip(link_preds, link_labels, labels):
            if crm_label == 0:# or len(reference) <= 5:
                continue
            cnt_golden += len(reference)
            cnt_pred += len(hypothesis)
            for pred, label in zip(hypothesis, reference):
                if label == -100 or pred == label:
                    cnt_cor_bi += 1
        prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
        f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
        return {'L_F1': f1_bi}

    model.eval()
    unwrapped_model = accelerator.unwrap_model(deepcopy(model))
    all_link_preds, all_link_labels, all_labels = [], [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, disable=not accelerator.is_local_main_process)
        for _, batch in pbar:
            inputs = prepare_inputs(batch, training=False)
            link_preds, link_labels = unwrapped_model(**inputs)
            all_link_preds += link_preds
            all_link_labels += link_labels
            all_labels += to_list(batch['labels'])

    # accelerator.print(list(zip(all_link_preds, all_link_labels)))
    result_dict = _cal_metrics(all_link_labels, all_link_preds, all_labels)

    return result_dict


def training(tokenizer):
    accelerator.print("Reading training records and examples for evaluation...")
    example_path_format = eval_path.replace(".json", "{}.tsv")
    example_postfix = "_negamode_{}".format(args.negative)
    if 0 < args.small < 100000:
        example_postfix += '_small_{}'.format(args.small//10)
    example_path = example_path_format.format(example_postfix)
    with open(example_path, "r", encoding='utf-8') as f:
        eval_example_lines = f.readlines()
    with open(eval_path, "r", encoding='utf-8') as f:
        eval_record_lines = f.readlines()
        if 0 < args.small < 100000:
            eval_record_lines = eval_record_lines[:args.small//10]
    accelerator.print("Reading is done!")

    accelerator.print("Reading training records and examples for training...")

    # CRM examples
    trunk_path = os.path.join(args.data_path, args.dataset, f"trunks_{args.trunk_size}")
    accelerator.print("Reading records and examples from {} for initialization...".format(trunk_path))
    with open(os.path.join(trunk_path, "0", "records.json"), "r", encoding='utf-8') as f:
        init_record_lines = f.readlines()
    with open(os.path.join(trunk_path, "0", "examples.txt"), "r", encoding='utf-8') as f:
        init_example_lines = f.readlines()

    # logging
    accelerator.print("Reading is done!")

    # expectation step initialization
    accelerator.print("Initializing addressees...")
    expectation_init(init_example_lines, init_record_lines)

    accelerator.wait_for_everyone()
    total_trunk_num = len(os.listdir(trunk_path))
    del init_example_lines, init_record_lines
    gc.collect()

    # training start
    train(total_trunk_num, eval_example_lines, eval_record_lines, tokenizer)
    accelerator.print("\n"+"="*128+"\n", "Training is done!!!!!!", "\n"+"="*128+"\n")


def main():
    set_seed()
    
    accelerator.print("Loading tokenizer and model...")
    tokenizer = tokenizer_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    tokenizer.truncation_side = 'left'

    accelerator.print("Pre-training process begins!")
    training(tokenizer)


if __name__ == "__main__":
    main()
