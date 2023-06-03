import os
import json
import torch
import argparse
import torch.utils.data as data
from tqdm import tqdm


# CONFIGS ===============================================================
USE_CUDA = True
METRICS = ['RL_F1', 'L_F1', 'RL_Prec', 'RL_Rec', 'L_Prec', 'L_Rec']


parser = argparse.ArgumentParser(description='Parameters for Discourse Parsing')

# storage path argument
parser.add_argument('--store_path', type=str, default='.', help="path for storage (output, checkpoints, caches, data, etc.)")

# training arguments
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=30)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='bert')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=384)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('-mun', '--max_utterance_num', type=int, default=14)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='molweni', choices=['stac', 'molweni'])
parser.add_argument('--logging_times', type=int, default=1, help='how many epochs to evaluate')
parser.add_argument('--cpu_count', type=int, default=16)
parser.add_argument('--mode', type=int, default=0, choices=[0, 1]) # 0: training, 1: evaluate

# arguments for models and data
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--model_file', type=str, default='baseline')

# arguments for loading pre-trained models
parser.add_argument('--pretrain_path', type=str, default='None')

args = parser.parse_args()

CPU_COUNT = args.cpu_count
RELATION2IDX = {'Comment': 0, 'Clarification_question': 1, 'Q-Elab': 2, 'QAP': 3, 'Result': 4, 'Acknowledgement': 5,\
     'Correction': 6, 'Continuation': 7, 'Parallel': 8, 'Elaboration': 9, 'Explanation': 10, 'Conditional': 11,\
     'Narration': 12, 'Background': 13, 'Contrast': 14, 'Alternation': 15, 'C': 16}

args.num_relations = len(RELATION2IDX)
args.data_path = os.path.join(args.store_path, args.data_path)
if "large" in args.model_name:
    args.save_path = args.save_path.replace("save", "lgsave")

args.save_root = '{}_saves'.format(args.dataset)
args.cache_root = 'caches'
args.save_root = os.path.join(args.store_path, args.save_root)
if 'None' not in args.pretrain_path:
    args.pretrain_path = os.path.join(args.store_path[:args.store_path.rfind('/')], args.pretrain_path)
    args.save_root += "_pretrained"
    if 'mlm' in args.pretrain_path:
        args.save_root += '_mlm'
    
args.save_path = args.save_root + '/' + args.model_type + '_{}_lr{}_'.format(args.model_file, args.learning_rate) + args.save_path
if args.store_path != ".":
    args.cache_root = os.path.join(args.store_path[:args.store_path.rfind('storage/')], 'storage', args.cache_root)
args.cache_path = args.cache_root + '/' + args.model_type + ('_large_' if 'large' in args.model_name else '_') + args.cache_path
args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
# END ===================================================================


class Dataset(data.Dataset):
    def __init__(self, examples, tokenizer, prefix, training=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.training = training

    def __getitem__(self, index):
        data_info = {}

        qqid, input_ids, attention_mask, token_type_ids, indicator_ids, sep_poses, link_label,\
             relation_label = convert_single_example(self.examples[index], self.tokenizer, training=self.training)

        data_info['qqids'] = qqid
        data_info['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        data_info['indicator_ids'] = torch.tensor(indicator_ids, dtype=torch.long)
        data_info['sep_poses'] = sep_poses
        data_info['link_labels'] = torch.tensor(link_label, dtype=torch.long) if link_label is not None else None
        data_info['relation_labels'] = torch.tensor(relation_label, dtype=torch.long) if relation_label is not None else None
        return data_info

    def __len__(self):
        return len(self.examples)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def collate_fn(data):
    data_info = {}
    float_type_keys = []
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = torch.stack(data_info[k])
        if isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                if not isinstance(data_info[k][0][id_key], torch.Tensor):
                    new_dict[id_key] = [sub_dict[id_key] for sub_dict in data_info[k]]
                    continue
                id_key_list = [torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float) for sub_dict in data_info[k]] # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = id_key_tensor
            data_info[k] = new_dict
    return data_info


def convert_single_example(example, tokenizer, training=True):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_sep_poses(input_ids):
        sep_poses = []
        last_idx = 0 # except the [CLS]
        for idx, inpidx in enumerate(input_ids):
            if inpidx == tokenizer.sep_token_id:
                sep_poses.append((last_idx+1, idx+1))
                last_idx = idx
        return sep_poses

    qqid = '{}-{}'.format(example["qid"], len(example['edus'])-1)
    utterances = [x['text'] for x in example['edus']]
    speakers = [x['speaker'] for x in example['edus']]
    spk2idx = {k: idx+1 for idx, k in enumerate(speakers)}
    speakers = ["#speaker{}#".format(spk2idx[k]) for k in speakers]
    relation_label = RELATION2IDX[example['relations'][-1]['type']]
    link_label = example['relations'][-1]['x']
    context = ''
    for speaker, utterance in list(zip(speakers[:-1], utterances[:-1])):
        context += tokenizer.sep_token + ' ' + speaker + ": " + utterance + ' '
    context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
    response = speaker[-1] + ": " + utterances[-1]
    try:
        ids_dict = tokenizer.encode_plus(context, response, padding='max_length',\
             truncation='only_first', max_length=args.max_length)
    except:
        ids_dict = tokenizer.encode_plus(context, response, padding='max_length',\
             truncation=True, max_length=args.max_length)
    input_ids = ids_dict["input_ids"]
    attention_mask = ids_dict["attention_mask"]
    token_type_ids = ids_dict["token_type_ids"]
    sep_poses = _get_sep_poses(input_ids)
    indicator_ids = [0] * len(input_ids) # make it all zero at this stage
    if training and link_label < len(sep_poses) - 1:
        for idx in range(sep_poses[link_label][0], sep_poses[link_label][1]):
            indicator_ids[idx] = 1 # training, teacher forcing
    assert len(sep_poses) == len(utterances)

    return qqid, input_ids, attention_mask, token_type_ids, indicator_ids, sep_poses, link_label, relation_label


def get_dataset(input_file, tokenizer, training=True, accelerator=None):
    postfix = ""
    for type_ in ["train", "valid", "test"]:
        if type_ in input_file:
            postfix = type_
            break
    
    log_fun = print if accelerator is None else accelerator.print
    log_fun("Reading examples for {}...".format(postfix))
    with open(input_file, "r", encoding='utf-8') as reader:
        examples = json.load(reader)["data"]["dialogues"]

    real_examples = []
    for idx, dial_dict in enumerate(examples):
        for rela_end in range(1, len(dial_dict["relations"])+1):
            max_utter_idx = dial_dict["relations"][rela_end-1]["y"]
            if dial_dict["relations"][rela_end-1]["y"] > max_utter_idx or dial_dict["relations"][rela_end-1]["x"] >= max_utter_idx or\
                 dial_dict["relations"][rela_end-1]["x"] == dial_dict["relations"][rela_end-1]["y"]: # eliminate invalid label or self-loop
                continue
            new_exp = {"edus": dial_dict["edus"][:max_utter_idx+1], "relations": dial_dict["relations"][:rela_end], "qid": f"{postfix}-{idx}"}
            real_examples.append(new_exp)
    examples = real_examples
    log_fun("Reading is done, total examples: {}".format(len(examples)))
    
    dataset = Dataset(examples, tokenizer, postfix, training=training)
    return dataset

    
if __name__ == "__main__":
    args.store_path = "/apdcephfs/share_916081/megnetoli/storage/ResponseSelectionMultiParty"
    input_file = os.path.join(args.store_path, "data/{}/test.json".format(args.dataset))

    from transformers import BartTokenizerFast, BertTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    print("Loading tokenizer...")
    # tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer.truncation_side = 'left'
    dataset = get_dataset(input_file, tokenizer, training=True)# if 'test' not in input_file else False)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8)
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        # print(batch["qid"][0])
        # print(batch["input_ids"].shape)
        # print(batch["attention_mask"].shape)
        # print(batch["labels"].shape)
        # print(batch["sep_poses"][0])
        pass