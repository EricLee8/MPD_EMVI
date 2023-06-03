import os
import json
import torch
import argparse
import torch.utils.data as data
from tqdm import tqdm


# CONFIGS ===============================================================
USE_CUDA = True
METRICS = ['auc', 'f1', 'acc', 'recall', 'precision']


parser = argparse.ArgumentParser(description='Parameters for Successful New Entry Prediction')

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
parser.add_argument('-ml', '--max_length', type=int, default=320)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='reddit', choices=['reddit', 'twitter'])
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

args.data_path = os.path.join(args.store_path, args.data_path)
if "large" in args.model_name:
    args.save_path = args.save_path.replace("save", "lgsave")
if args.model_type == "electra":
    args.adam_epsilon = 1e-6

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

        input_ids, attention_mask, token_type_ids, indicator_ids, sep_poses,\
             label = convert_single_example(self.examples[index], self.tokenizer, training=self.training)

        data_info['qid_tensor'] = torch.tensor(index, dtype=torch.long)
        data_info['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        data_info['indicator_ids'] = torch.tensor(indicator_ids, dtype=torch.long)
        data_info['sep_poses'] = sep_poses
        data_info['labels'] = torch.tensor(label, dtype=torch.float) if label is not None else None
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

    utter_info_list, label = example[0], example[1]
    speakers, utterances = [], []
    for utter_info in utter_info_list:
        speakers.append(utter_info[2])
        utterances.append(utter_info[3])
    spk2idx = {k: idx+1 for idx, k in enumerate(speakers)}
    speakers = ["#speaker{}#".format(spk2idx[k]) for k in speakers]

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
    s, e = sep_poses[-2] # took the last turn as addressee
    for idx in range(s, e):
        indicator_ids[idx] = 1


    return input_ids, attention_mask, token_type_ids, indicator_ids, sep_poses, label


def get_dataset(input_file, tokenizer, training=True, accelerator=None):
    postfix = ""
    for type_ in ["train", "valid", "test"]:
        if type_ in input_file:
            postfix = type_
            break
    
    log_fun = print if accelerator is None else accelerator.print
    log_fun("Reading examples for {}...".format(postfix))
    with open(input_file, "r", encoding='utf-8') as reader:
        lines = reader.readlines()

    examples = []
    for line in lines:
        examples.append(json.loads(line.strip()))
    log_fun("Reading is done, total examples: {}".format(len(examples)))

    dataset = Dataset(examples, tokenizer, postfix, training=training)
    return dataset

    
if __name__ == "__main__":
    pass
